import os
import glob
import math
import re

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.utils.data as data
from Bio.SeqIO.FastaIO import SimpleFastaParser

from utils import temp_seed

PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY*'
PROTEIN_REORDERED_ALPHABET = 'DEKRHNQSTPGAVILMCFYW*'
RNA_ALPHABET = 'ACGU*'
DNA_ALPHABET = 'ACGT*'
START_END = "*"


def get_alphabet(alphabet_type='protein'):
    if alphabet_type == 'protein':
        return PROTEIN_ALPHABET, PROTEIN_REORDERED_ALPHABET
    elif alphabet_type == 'RNA':
        return RNA_ALPHABET, RNA_ALPHABET
    elif alphabet_type == 'DNA':
        return DNA_ALPHABET, DNA_ALPHABET
    else:
        raise ValueError('unknown alphabet type')


class GeneratorDataset(data.Dataset):
    """A Dataset that can be used as a generator."""
    def __init__(
            self,
            batch_size=32,
            unlimited_epoch=True,
    ):
        self.batch_size = batch_size
        self.unlimited_epoch = unlimited_epoch

    @property
    def params(self):
        return {"batch_size": self.batch_size, "unlimited_epoch": self.unlimited_epoch}

    @params.setter
    def params(self, d):
        if 'batch_size' in d:
            self.batch_size = d['batch_size']
        if 'unlimited_epoch' in d:
            self.unlimited_epoch = d['unlimited_epoch']

    @property
    def n_eff(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.unlimited_epoch:
            return 2 ** 62
        else:
            return math.ceil(self.n_eff / self.batch_size)

    @staticmethod
    def collate_fn(batch):
        return batch[0]


class GeneratorDataLoader(data.DataLoader):
    """A DataLoader used with a GeneratorDataset"""
    def __init__(self, dataset: GeneratorDataset, **kwargs):
        kwargs.update(dict(
            batch_size=1, shuffle=False, sampler=None, batch_sampler=None, collate_fn=dataset.collate_fn,
        ))
        super(GeneratorDataLoader, self).__init__(
            dataset,
            **kwargs)


class TrainTestDataset(data.Dataset):
    """A Dataset that has training and testing modes"""
    def __init__(self):
        self._training = True

    def train(self, training=True):
        self._training = training

    def test(self):
        self.train(False)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TrainValTestDataset(data.Dataset):
    """A Dataset that has training, validation, and testing modes"""
    def __init__(self):
        self._mode = 'train'

    def train(self, mode='train'):
        self._mode = mode

    def val(self):
        self.train('val')

    def test(self):
        self.train('test')

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def sequences_to_decoder_onehot(
        sequences, input_char_map, output_char_map, reverse=False, matching=False, start_char='*', end_char='*'):
    num_seqs = len(sequences)
    max_seq_len = max([len(seq) for seq in sequences]) + 1
    decoder_input = np.zeros((num_seqs, len(input_char_map), 1, max_seq_len))
    decoder_output = np.zeros((num_seqs, len(output_char_map), 1, max_seq_len))
    decoder_mask = np.zeros((num_seqs, 1, 1, max_seq_len))

    if matching:
        decoder_input_r = np.zeros((num_seqs, len(input_char_map), 1, max_seq_len))
        decoder_output_r = np.zeros((num_seqs, len(output_char_map), 1, max_seq_len))
    else:
        decoder_input_r = None
        decoder_output_r = None

    for i, sequence in enumerate(sequences):
        if reverse:
            sequence = sequence[::-1]

        decoder_input_seq = start_char + sequence
        decoder_output_seq = sequence + end_char

        if matching:
            sequence_r = sequence[::-1]
            decoder_input_seq_r = start_char + sequence_r
            decoder_output_seq_r = sequence_r + end_char
        else:
            decoder_input_seq_r = None
            decoder_output_seq_r = None

        for j in range(len(decoder_input_seq)):
            decoder_input[i, input_char_map[decoder_input_seq[j]], 0, j] = 1
            decoder_output[i, output_char_map[decoder_output_seq[j]], 0, j] = 1
            decoder_mask[i, 0, 0, j] = 1

            if matching:
                decoder_input_r[i, input_char_map[decoder_input_seq_r[j]], 0, j] = 1
                decoder_output_r[i, output_char_map[decoder_output_seq_r[j]], 0, j] = 1

    return decoder_input, decoder_output, decoder_mask, decoder_input_r, decoder_output_r


def sequences_to_encoder_onehot(sequences, char_map, start_char='', end_char=''):
    num_seqs = len(sequences)
    max_seq_len = max([len(seq) for seq in sequences]) + len(start_char) + len(end_char)
    encoder_input = np.zeros((num_seqs, len(char_map), 1, max_seq_len))
    encoder_mask = np.zeros((num_seqs, 1, 1, max_seq_len))

    for i, sequence in enumerate(sequences):
        encoder_input_seq = start_char + sequence + end_char

        for j in range(len(encoder_input_seq)):
            encoder_input[i, char_map[encoder_input_seq[j]], 0, j] = 1
            encoder_mask[i, 0, 0, j] = 1

    return encoder_input, encoder_mask


class SequenceDataset(GeneratorDataset):
    """Abstract sequence dataset"""
    SUPPORTED_OUTPUT_SHAPES = ['NCHW', 'NHWC', 'NLC']

    def __init__(
            self,
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder,encoder',
    ):
        super(SequenceDataset, self).__init__(batch_size=batch_size, unlimited_epoch=unlimited_epoch)

        self.alphabet_type = alphabet_type
        self.reverse = reverse
        self.matching = matching
        self.output_shape = output_shape
        self.output_types = output_types
        self.max_seq_len = -1

        if output_shape not in self.SUPPORTED_OUTPUT_SHAPES:
            raise KeyError(f'Unsupported output shape: {output_shape}')

        self.aa_dict = self.idx_to_aa = self.output_aa_dict = self.output_idx_to_aa = None
        self.update_aa_dict()

    @property
    def params(self):
        params = super(SequenceDataset, self).params
        params.update({
            "alphabet_type": self.alphabet_type,
            "reverse": self.reverse,
            "matching": self.matching,
            "output_shape": self.output_shape,
            "output_types": self.output_types,
        })
        return params

    @params.setter
    def params(self, d):
        GeneratorDataset.params.__set__(self, d)
        if 'alphabet_type' in d:
            self.alphabet_type = d['alphabet_type']
            self.update_aa_dict()
        if 'reverse' in d:
            self.reverse = d['reverse']
        if 'matching' in d:
            self.matching = d['matching']
        if 'output_shape' in d:
            self.output_shape = d['output_shape']
        if 'output_types' in d:
            self.output_types = d['output_types']

    @property
    def alphabet(self):
        if self.alphabet_type == 'protein':
            return PROTEIN_ALPHABET
        elif self.alphabet_type == 'RNA':
            return RNA_ALPHABET
        elif self.alphabet_type == 'DNA':
            return DNA_ALPHABET

    @property
    def output_alphabet(self):
        return self.alphabet

    def update_aa_dict(self):
        self.aa_dict = {aa: i for i, aa in enumerate(self.alphabet)}
        self.idx_to_aa = {i: aa for i, aa in enumerate(self.alphabet)}
        self.output_aa_dict = {aa: i for i, aa in enumerate(self.output_alphabet)}
        self.output_idx_to_aa = {i: aa for i, aa in enumerate(self.output_alphabet)}

    @property
    def n_eff(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def sequences_to_onehot(self, sequences, reverse=None, matching=None):
        """

        :param sequences: list/iterable of strings
        :param reverse: reverse the sequences
        :param matching: output forward and reverse sequences
        :return: dictionary of strings
        """
        reverse = self.reverse if reverse is None else reverse
        matching = self.matching if matching is None else matching
        output = {}

        if 'decoder' in self.output_types:
            decoder_input, decoder_output, decoder_mask, decoder_input_r, decoder_output_r = \
                sequences_to_decoder_onehot(sequences, self.aa_dict, self.output_aa_dict,
                                            reverse=reverse, matching=matching)

            if matching:
                output.update({
                    'decoder_input': decoder_input,
                    'decoder_output': decoder_output,
                    'decoder_mask': decoder_mask,
                    'decoder_input_r': decoder_input_r,
                    'decoder_output_r': decoder_output_r
                })
            else:
                output.update({
                    'decoder_input': decoder_input,
                    'decoder_output': decoder_output,
                    'decoder_mask': decoder_mask
                })
        if 'encoder' in self.output_types:
            encoder_input, encoder_mask = sequences_to_encoder_onehot(sequences, self.aa_dict)
            output.update({
                'encoder_input': encoder_input,
                'encoder_mask': encoder_mask
            })

        for key in output.keys():
            output[key] = torch.as_tensor(output[key], dtype=torch.float32)
            if self.output_shape == 'NHWC':
                output[key] = output[key].permute(0, 2, 3, 1).contiguous()
            elif self.output_shape == 'NLC':
                output[key] = output[key].squeeze(2).permute(0, 2, 1).contiguous()

        return output


class FastaDataset(SequenceDataset):
    """Load batches of sequences from a fasta file, either sequentially or sampled isotropically"""

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=False,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
            # TODO add shuffle parameter: iterate through shuffled sequences
    ):
        super(FastaDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.names = None
        self.sequences = None

        self.load_data()

    def load_data(self):
        filename = os.path.join(self.working_dir, self.dataset)
        names_list = []
        sequence_list = []
        max_seq_len = 0

        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue

                names_list.append(title)
                sequence_list.append(seq)
                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)

        self.names = np.array(names_list)
        self.sequences = np.array(sequence_list)
        self.max_seq_len = max_seq_len

        print("Number of sequences:", self.n_eff)
        print("Max sequence length:", max_seq_len)

    @property
    def n_eff(self):
        return len(self.sequences)  # not a true n_eff

    def __getitem__(self, index):
        """
        :param index: batch index; ignored if unlimited_epoch
        :return: batch of size self.batch_size
        """

        if self.unlimited_epoch:
            indices = np.random.randint(0, self.n_eff, self.batch_size)
        else:
            first_index = index * self.batch_size
            last_index = min((index+1) * self.batch_size, self.n_eff)
            indices = np.arange(first_index, last_index)

        seqs = self.sequences[indices]
        batch = self.sequences_to_onehot(seqs)
        batch['names'] = self.names[indices]
        batch['sequences'] = seqs
        return batch


class SingleFamilyDataset(SequenceDataset):
    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
    ):
        super(SingleFamilyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.family_name_to_sequence_list = {}
        self.family_name_to_sequence_weight_list = {}
        self.family_name_to_n_eff = {}
        self.family_name_list = []
        self.family_idx_list = []
        self.family_name = ''
        self.family_name_to_idx = {}
        self.idx_to_family_name = {}

        self.num_families = 0
        self.max_family_size = 0

        self.load_data()

    def load_data(self):
        max_seq_len = 0
        max_family_size = 0
        family_name = ''
        weight_list = []

        f_names = glob.glob(self.working_dir + '/datasets/sequences/' + self.dataset + '*.fa')
        if len(f_names) != 1:
            raise AssertionError('Wrong number of families: {}'.format(len(f_names)))

        for filename in f_names:
            sequence_list = []
            weight_list = []

            family_name_list = filename.split('/')[-1].split('_')
            family_name = family_name_list[0] + '_' + family_name_list[1]
            print(family_name)

            family_size = 0
            ind_family_idx_list = []
            with open(filename, 'r') as fa:
                for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                    weight = float(title.split(':')[-1])
                    valid = True
                    for letter in seq:
                        if letter not in self.aa_dict:
                            valid = False
                    if not valid:
                        continue

                    sequence_list.append(seq)
                    ind_family_idx_list.append(family_size)
                    weight_list.append(weight)
                    family_size += 1
                    if len(seq) > max_seq_len:
                        max_seq_len = len(seq)

            if family_size > max_family_size:
                max_family_size = family_size

            self.family_name_to_sequence_list[family_name] = sequence_list
            self.family_name_to_sequence_weight_list[family_name] = (
                np.asarray(weight_list) / np.sum(weight_list)
            ).tolist()
            self.family_name_to_n_eff[family_name] = np.sum(weight_list)
            self.family_name = family_name
            self.family_name_list.append(family_name)
            self.family_idx_list.append(ind_family_idx_list)

        self.family_name = family_name
        self.max_seq_len = max_seq_len
        self.num_families = len(self.family_name_list)
        self.max_family_size = max_family_size

        print("Number of families:", self.num_families)
        print("Neff:", np.sum(weight_list))
        print("Max family size:", max_family_size)
        print("Max sequence length:", max_seq_len)

        for i, family_name in enumerate(self.family_name_list):
            self.family_name_to_idx[family_name] = i
            self.idx_to_family_name[i] = family_name

    @property
    def n_eff(self):
        return self.family_name_to_n_eff[self.family_name]

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        family_name = self.family_name
        family_seqs = self.family_name_to_sequence_list[family_name]
        family_weights = self.family_name_to_sequence_weight_list[family_name]

        seq_idx = np.random.choice(len(family_seqs), self.batch_size, p=family_weights)
        seqs = [family_seqs[idx] for idx in seq_idx]

        batch = self.sequences_to_onehot(seqs)
        return batch


class SingleClusteredSequenceDataset(SequenceDataset):
    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
    ):
        super(SingleClusteredSequenceDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.name_to_sequence = {}
        self.clu1_to_seq_names = {}
        self.clu1_list = []

        self.load_data()

    def load_data(self):
        max_seq_len = 0
        num_seqs = 0
        filename = os.path.join(self.working_dir, self.dataset)
        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                name, clu1 = title.split(':')  # TODO handle different split lengths
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue

                self.name_to_sequence[name] = seq
                if clu1 in self.clu1_to_seq_names:
                    self.clu1_to_seq_names[clu1].append(name)
                else:
                    self.clu1_to_seq_names[clu1] = [name]

                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)
                num_seqs += 1

        self.clu1_list = list(self.clu1_to_seq_names.keys())
        self.max_seq_len = max_seq_len

        print("Num clusters:", len(self.clu1_list))
        print("Num sequences:", num_seqs)
        print("Max sequence length:", max_seq_len)

    @property
    def n_eff(self):
        return len(self.clu1_list)

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        names = []
        seqs = []
        for i in range(self.batch_size):
            # Pick a cluster id90
            clu1_idx = np.random.randint(0, len(self.clu1_list))
            clu1 = self.clu1_list[clu1_idx]

            # Then pick a random sequence all in those clusters
            seq_name = np.random.choice(self.clu1_to_seq_names[clu1])
            names.append(seq_name)

            # then grab the associated sequence
            seqs.append(self.name_to_sequence[seq_name])

        batch = self.sequences_to_onehot(seqs)
        batch['names'] = names
        batch['sequences'] = seqs
        return batch


class DoubleClusteredSequenceDataset(SequenceDataset):
    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
    ):
        super(DoubleClusteredSequenceDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.name_to_sequence = {}
        self.clu1_to_clu2_to_seq_names = {}
        self.clu1_list = []

        self.load_data()

    def load_data(self):
        max_seq_len = 0
        num_subclusters = 0
        num_seqs = 0
        filename = os.path.join(self.working_dir, self.dataset)
        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                name, clu1, clu2 = title.split(':')
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue

                self.name_to_sequence[name] = seq
                if clu1 in self.clu1_to_clu2_to_seq_names:
                    if clu2 in self.clu1_to_clu2_to_seq_names[clu1]:
                        self.clu1_to_clu2_to_seq_names[clu1][clu2].append(name)
                    else:
                        self.clu1_to_clu2_to_seq_names[clu1][clu2] = [name]
                        num_subclusters += 1
                else:
                    self.clu1_to_clu2_to_seq_names[clu1] = {clu2: [name]}
                    num_subclusters += 1

                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)
                num_seqs += 1

        self.clu1_list = list(self.clu1_to_clu2_to_seq_names.keys())
        self.max_seq_len = max_seq_len

        print("Num clusters:", len(self.clu1_list))
        print("Num subclusters:", num_subclusters)
        print("Num sequences:", num_seqs)
        print("Max sequence length:", max_seq_len)

    @property
    def n_eff(self):
        return len(self.clu1_list)

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        names = []
        seqs = []
        for i in range(self.batch_size):
            # Pick a cluster id80
            clu1_idx = np.random.randint(0, len(self.clu1_list))
            clu1 = self.clu1_list[clu1_idx]

            # Then pick a cluster id90 from the cluster id80s
            clu2 = np.random.choice(list(self.clu1_to_clu2_to_seq_names[clu1].keys()))

            # Then pick a random sequence all in those clusters
            seq_name = np.random.choice(self.clu1_to_clu2_to_seq_names[clu1][clu2])
            names.append(seq_name)

            # then grab the associated sequence
            seqs.append(self.name_to_sequence[seq_name])

        batch = self.sequences_to_onehot(seqs)
        batch['names'] = names
        batch['sequences'] = seqs
        return batch


class IndexedFastaDataset(SequenceDataset):
    """Load batches of sequences from an indexed fasta file, either sequentially or sampled isotropically"""

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=False,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
            # TODO add shuffle parameter: iterate through shuffled sequences
    ):
        super(IndexedFastaDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
        )
        self.dataset = dataset
        self.dataset_f = os.path.join(working_dir, dataset)
        self.dataset_idx = dataset + '.fai'
        self.working_dir = working_dir

        self.names = None
        self.offsets = None

        self.load_data()

    def load_data(self):
        filename = os.path.join(self.working_dir, self.dataset_idx)
        num_seqs = 0
        max_name_len = 0
        max_seq_len = 0
        with open(filename, 'rb') as f:
            for line in f:
                line = line.split(b'\t')
                name_len, seq_len = len(line[0]), int(line[1])
                num_seqs += 1
                if name_len > max_name_len:
                    max_name_len = name_len
                if seq_len > max_seq_len:
                    max_seq_len = seq_len

            self.names = np.empty(num_seqs, dtype=f'<U{max_name_len}')
            self.offsets = np.zeros(num_seqs, dtype=np.int)

            f.seek(0)
            for i, line in enumerate(f):
                line = line.split(b'\t')
                self.names[i], self.offsets[i] = line[0], int(line[2])

        self.max_seq_len = max_seq_len

        print("Num sequences:", len(self.offsets))
        print("Max sequence length:", max_seq_len)

    @property
    def n_eff(self):
        return len(self.offsets)  # not a true n_eff

    def __getitem__(self, index):
        """
        :param index: batch index; ignored if unlimited_epoch
        :return: batch of size self.batch_size
        """

        if self.unlimited_epoch:
            indices = np.random.randint(0, self.n_eff, self.batch_size)
        else:
            first_index = index * self.batch_size
            last_index = min((index+1) * self.batch_size, self.n_eff)
            indices = np.arange(first_index, last_index)

        offsets = self.offsets[indices]
        seqs = []
        with open(self.dataset_f, 'rt', buffering=1) as dataset_f:
            for offset in offsets:
                dataset_f.seek(offset)
                seqs.append(dataset_f.readline().rstrip())
        seqs = np.array(seqs)

        try:
            batch = self.sequences_to_onehot(seqs)
            batch['names'] = self.names[indices]
            batch['sequences'] = seqs
            return batch
        except KeyError as e:
            print(offsets, seqs, flush=True)
            raise e


class SingleClusteredIndexedSequenceDataset(SequenceDataset):
    """
    Reads an indexed fasta dataset.
    Index filename is (fasta_file).fai
    Create index using `samtools faidx (fasta file)`.
    """

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
    ):
        super(SingleClusteredIndexedSequenceDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
        )
        self.dataset = dataset
        self.dataset_f = os.path.join(working_dir, dataset)
        self.dataset_idx = dataset + '.fai'
        self.working_dir = working_dir

        self.clu1_to_offset = {}
        self.clu1_list = []

        self.load_data()

    def load_data(self):
        filename = os.path.join(self.working_dir, self.dataset_idx)
        max_seq_len = 0
        num_seqs = 0
        with open(filename, 'rb') as f:
            for line in f:
                line = line.split(b'\t')
                title, length, offset = line[0], int(line[1]), int(line[2])
                name, clu1, clu2 = title.split(b':')

                if clu1 in self.clu1_to_offset:
                    self.clu1_to_offset[clu1].append(offset)
                else:
                    self.clu1_to_offset[clu1] = [offset]

                if length > max_seq_len:
                    max_seq_len = length
                num_seqs += 1

        self.clu1_list = list(self.clu1_to_offset.keys())
        self.max_seq_len = max_seq_len

        print("Num clusters:", len(self.clu1_list))
        print("Num sequences:", num_seqs)
        print("Max sequence length:", max_seq_len)

    @property
    def n_eff(self):
        return len(self.clu1_list)

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        offsets = []
        seqs = []

        with open(self.dataset_f, 'rt', buffering=1) as dataset_f:
            for i in range(self.batch_size):
                # Pick a cluster id80
                clu1_idx = np.random.randint(0, len(self.clu1_list))
                clu1 = self.clu1_list[clu1_idx]

                # Then pick a cluster id90 from the cluster id80s
                clu2 = np.random.choice(list(self.clu1_to_offset[clu1].keys()))

                # Then pick a random sequence all in those clusters
                offset = np.random.choice(self.clu1_to_offset[clu1][clu2])
                offsets.append(offset)

                # then grab the associated sequence
                dataset_f.seek(offset)
                seqs.append(dataset_f.readline().rstrip())

        try:
            batch = self.sequences_to_onehot(seqs)
            return batch
        except KeyError as e:
            print(offsets, seqs, flush=True)
            raise e


class DoubleClusteredIndexedSequenceDataset(SequenceDataset):
    """
    Reads an indexed fasta dataset.
    Index filename is (fasta_file).fai
    Create index using `samtools faidx (fasta file)`.
    """

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
    ):
        super(DoubleClusteredIndexedSequenceDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
        )
        self.dataset = dataset
        self.dataset_f = os.path.join(working_dir, dataset)
        self.dataset_idx = dataset + '.fai'
        self.working_dir = working_dir

        self.clu1_to_clu2_to_offset = {}
        self.clu1_list = []

        self.load_data()

    def load_data(self):
        filename = os.path.join(self.working_dir, self.dataset_idx)
        max_seq_len = 0
        num_subclusters = 0
        num_seqs = 0
        with open(filename, 'rb') as f:
            for line in f:
                line = line.split(b'\t')
                title, length, offset = line[0], int(line[1]), int(line[2])
                name, clu1, clu2 = title.split(b':')

                if clu1 in self.clu1_to_clu2_to_offset:
                    if clu2 in self.clu1_to_clu2_to_offset[clu1]:
                        self.clu1_to_clu2_to_offset[clu1][clu2].append(offset)
                    else:
                        self.clu1_to_clu2_to_offset[clu1][clu2] = [offset]
                        num_subclusters += 1
                else:
                    self.clu1_to_clu2_to_offset[clu1] = {clu2: [offset]}
                    num_subclusters += 1

                if length > max_seq_len:
                    max_seq_len = length
                num_seqs += 1

        self.clu1_list = list(self.clu1_to_clu2_to_offset.keys())
        self.max_seq_len = max_seq_len

        print("Num clusters:", len(self.clu1_list))
        print("Num subclusters:", num_subclusters)
        print("Num sequences:", num_seqs)
        print("Max sequence length:", max_seq_len)

    @property
    def n_eff(self):
        return len(self.clu1_list)

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        offsets = []
        seqs = []

        with open(self.dataset_f, 'rt', buffering=1) as dataset_f:
            for i in range(self.batch_size):
                # Pick a cluster id80
                clu1_idx = np.random.randint(0, len(self.clu1_list))
                clu1 = self.clu1_list[clu1_idx]

                # Then pick a cluster id90 from the cluster id80s
                clu2 = np.random.choice(list(self.clu1_to_clu2_to_offset[clu1].keys()))

                # Then pick a random sequence all in those clusters
                offset = np.random.choice(self.clu1_to_clu2_to_offset[clu1][clu2])
                offsets.append(offset)

                # then grab the associated sequence
                dataset_f.seek(offset)
                seqs.append(dataset_f.readline().rstrip())

        try:
            batch = self.sequences_to_onehot(seqs)
            return batch
        except KeyError as e:
            print(offsets, seqs, flush=True)
            raise e


class AntibodySequenceDataset(SequenceDataset):
    IPI_VL_SEQS = ['VK1-39', 'VL1-51', 'VK3-15']
    IPI_VH_SEQS = ['VH1-46', 'VH1-69', 'VH3-7', 'VH3-15', 'VH4-39', 'VH5-51']
    LABELED = False

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='encoder',
            include_vl=False,
            include_vh=False,
    ):
        SequenceDataset.__init__(
            self,
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
        )
        self.dataset = dataset
        self.working_dir = working_dir
        self.include_vl = include_vl
        self.include_vh = include_vh

        self.vl_list = self.IPI_VL_SEQS.copy()
        self.vh_list = self.IPI_VH_SEQS.copy()

    @property
    def light_to_idx(self):
        if self.vh_list is None:
            raise RuntimeError("VL list not loaded.")
        else:
            return {vh: i for i, vh in enumerate(self.vl_list)}

    @property
    def heavy_to_idx(self):
        if self.vh_list is None:
            raise RuntimeError("VH list not loaded.")
        else:
            return {vh: i for i, vh in enumerate(self.vh_list)}

    @property
    def input_dim(self):
        input_dim = len(self.alphabet)
        if self.include_vl:
            input_dim += len(self.light_to_idx)
        if self.include_vh:
            input_dim += len(self.heavy_to_idx)
        return input_dim

    @property
    def params(self):
        params = super(AntibodySequenceDataset, self).params
        params.update({
            "include_vl": self.include_vl,
            "include_vh": self.include_vh,
            "vl_seqs": self.vl_list,
            "vh_seqs": self.vh_list,
        })
        return params

    @params.setter
    def params(self, d):
        if 'for_decoder' in d:
            if d['for_decoder']:
                d['output_types'] = 'decoder'
            else:
                d['output_types'] = 'encoder'
        SequenceDataset.params.__set__(self, d)
        if 'include_vl' in d:
            self.include_vl = d['include_vl']
        if 'include_vh' in d:
            self.include_vh = d['include_vh']
        if 'vl_seqs' in d:
            self.vl_list = d['vl_seqs']
        if 'vh_seqs' in d:
            self.vh_list = d['vh_seqs']

    def sequences_to_onehot(self, sequences, vls=None, vhs=None, reverse=None, matching=None):
        reverse = self.reverse if reverse is None else reverse
        matching = self.matching if matching is None else matching
        num_seqs = len(sequences)
        for i in range(num_seqs):
            # normalize CDR3 sequences to exclude constant characters
            # TODO add strip_cw param
            if sequences[i][0] == 'C':
                sequences[i] = sequences[i][1:]
            if sequences[i][-1] == 'W':
                sequences[i] = sequences[i][:-1]

        if 'decoder' in self.output_types:
            seq_arr, seq_output_arr, seq_mask, seq_arr_r, seq_output_arr_r = sequences_to_decoder_onehot(
                sequences, self.aa_dict, self.output_aa_dict, reverse=reverse, matching=matching
            )
        else:
            seq_arr, seq_mask = sequences_to_encoder_onehot(sequences, self.aa_dict)
            seq_output_arr = seq_arr_r = seq_output_arr_r = None

        light_arr = heavy_arr = None
        if self.include_vl:
            light_arr = np.zeros((num_seqs, len(self.light_to_idx), 1, seq_arr.shape[-1]))
        if self.include_vh:
            heavy_arr = np.zeros((num_seqs, len(self.heavy_to_idx), 1, seq_arr.shape[-1]))

        for i in range(num_seqs):
            if self.include_vl:
                light_arr[i, self.light_to_idx[vls[i]], 0, :] = 1.
            if self.include_vl:
                heavy_arr[i, self.heavy_to_idx[vhs[i]], 0, :] = 1.

        if self.include_vl:
            seq_arr = np.concatenate((seq_arr, light_arr), axis=1)
            if seq_arr_r is not None:
                seq_arr_r = np.concatenate((seq_arr_r, light_arr), axis=1)
        if self.include_vh:
            seq_arr = np.concatenate((seq_arr, heavy_arr), axis=1)
            if seq_arr_r is not None:
                seq_arr_r = np.concatenate((seq_arr_r, heavy_arr), axis=1)

        if 'decoder' in self.output_types:
            output = {
                'decoder_input': seq_arr,
                'decoder_mask': seq_mask,
                'decoder_output': seq_output_arr,
                'decoder_input_r': seq_arr_r,
                'decoder_output_r': seq_output_arr_r,
            }
        else:
            output = {'encoder_input': seq_arr, 'encoder_mask': seq_mask}
        for key in output.keys():
            output[key] = torch.as_tensor(output[key], dtype=torch.float32)
            if self.output_shape == 'NHWC':
                output[key] = output[key].permute(0, 2, 3, 1).contiguous()
            elif self.output_shape == 'NLC':
                output[key] = output[key].squeeze(2).permute(0, 2, 1).contiguous()
        return output

    @property
    def n_eff(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class IPIFastaDataset(AntibodySequenceDataset):
    """Unweighted antibody dataset.
    fasta: >*_heavy-{VH}_light-{VL}*
    """

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
            include_vl=False,
            include_vh=False,
    ):
        AntibodySequenceDataset.__init__(
            self,
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
            include_vl=include_vl,
            include_vh=include_vh,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.names = None
        self.sequences = None
        self.vhs = None
        self.vls = None

        self.load_data()

    def load_data(self):
        filename = os.path.join(self.working_dir, self.dataset)
        names_list = []
        sequence_list = []
        vhs = []
        vls = []
        max_seq_len = 0
        skipped_seqs = 0
        name_pat = re.compile(r'_heavy-([A-Z0-9\-]+)_light-([A-Z0-9\-]+)')

        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    skipped_seqs += 1
                    continue

                match = name_pat.search(title)
                if not match or match.group(1) not in self.vh_list or match.group(2) not in self.vl_list:
                    skipped_seqs += 1
                    continue

                names_list.append(title)
                sequence_list.append(seq)
                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)
                vhs.append(match.group(1))
                vls.append(match.group(2))

        self.names = np.array(names_list)
        self.sequences = np.array(sequence_list)
        self.max_seq_len = max_seq_len
        self.vhs = np.array(vhs)
        self.vls = np.array(vls)

        print("Number of sequences:", self.n_eff)
        print("Number of sequences skipped:", skipped_seqs)
        print("Max sequence length:", max_seq_len)

    @property
    def n_eff(self):
        return len(self.sequences)  # not a true n_eff

    def __getitem__(self, index):
        """
        :param index: batch index; ignored if unlimited_epoch
        :return: batch of size self.batch_size
        """

        if self.unlimited_epoch:
            indices = np.random.randint(0, self.n_eff, self.batch_size)
        else:
            first_index = index * self.batch_size
            last_index = min((index + 1) * self.batch_size, self.n_eff)
            indices = np.arange(first_index, last_index)

        seqs = self.sequences[indices]
        names = self.names[indices]
        vhs = self.vhs[indices]
        vls = self.vls[indices]
        batch = self.sequences_to_onehot(seqs, vhs=vhs, vls=vls)
        batch['names'] = names
        batch['sequences'] = seqs
        return batch


class IPISingleClusteredSequenceDataset(AntibodySequenceDataset):
    """Single-weighted antibody dataset.
    fasta: >seq:vh:vl:clu1
    clu1: cluster id
    """

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
            include_vl=False,
            include_vh=False,
    ):
        AntibodySequenceDataset.__init__(
            self,
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
            include_vl=include_vl,
            include_vh=include_vh,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.name_to_sequence = {}
        self.clu1_to_seq_names = {}
        self.clu1_list = []

        self.load_data()

    def load_data(self):
        max_seq_len = 0
        num_seqs = 0
        filename = os.path.join(self.working_dir, self.dataset)
        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                name, vh, vl, clu1 = title.split(':')
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue
                elif vh not in self.vh_list:
                    print(f"Unrecognized VH gene: {vh}")
                    continue
                elif vl not in self.vl_list:
                    print(f"Unrecognized VL gene: {vl}")
                    continue

                name = f"{name}:{vh}:{vl}"
                if name in self.name_to_sequence:
                    print(f"Name collision: {name}")
                self.name_to_sequence[name] = seq
                if clu1 in self.clu1_to_seq_names:
                    self.clu1_to_seq_names[clu1].append(name)
                else:
                    self.clu1_to_seq_names[clu1] = [name]

                if len(seq) > max_seq_len:
                    max_seq_len = len(seq)
                num_seqs += 1

        self.clu1_list = list(self.clu1_to_seq_names.keys())
        self.max_seq_len = max_seq_len

        print("Num clusters:", len(self.clu1_list))
        print("Num sequences:", num_seqs)
        print("Max sequence length:", max_seq_len)

    @property
    def n_eff(self):
        return len(self.clu1_list)

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        names = []
        vhs = []
        vls = []
        seqs = []
        for i in range(self.batch_size):
            # Pick a cluster id90
            clu1_idx = np.random.randint(0, len(self.clu1_list))
            clu1 = self.clu1_list[clu1_idx]

            # Then pick a random sequence all in those clusters
            seq_name = np.random.choice(self.clu1_to_seq_names[clu1])
            _, vh, vl = seq_name.split(':')
            names.append(seq_name)
            vhs.append(vh)
            vls.append(vl)

            # then grab the associated sequence
            seqs.append(self.name_to_sequence[seq_name])

        batch = self.sequences_to_onehot(seqs, vhs=vhs, vls=vls)
        batch['names'] = names
        batch['sequences'] = seqs
        return batch


class IPITrainTestDataset(AntibodySequenceDataset, TrainTestDataset):
    LABELED = True

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='encoder',
            comparisons=(('Aff1', 'PSR1', 0., 0.),),  # before, after, thresh_before, thresh_after
            train_test_split=1.0,
            split_seed=42,
            include_vl=False,
            include_vh=False,
    ):
        AntibodySequenceDataset.__init__(
            self,
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
            include_vl=include_vl,
            include_vh=include_vh,
        )
        TrainTestDataset.__init__(self)
        self.dataset = dataset
        self.working_dir = working_dir
        self.comparisons = comparisons
        self.train_test_split = train_test_split
        self.split_seed = split_seed
        self.include_vl = include_vl
        self.include_vh = include_vh

        self.cdr_to_output = {}
        self.cdr_to_heavy = {}
        self.cdr_to_light = {}
        self.all_cdr_seqs = []
        self.cdr_seqs_train = []
        self.cdr_seqs_test = []
        self.comparison_pos_weights = torch.ones(len(comparisons))

        self.load_data()

    def load_data(self):
        seq_col = 'CDR3'
        heavy_col = 'heavy'
        light_col = 'light'
        count_cols = list({col: None for comparison in self.comparisons for col in comparison[:2]}.keys())

        # load data file
        filename = os.path.join(self.working_dir, self.dataset)
        use_cols = [seq_col, heavy_col, light_col] + count_cols
        df = pd.read_csv(filename, usecols=use_cols)
        df[count_cols] = df[count_cols].fillna(0.)

        # load output data
        comparison_cdr_to_output = []
        for i_comparison, comparison in enumerate(self.comparisons):
            before, after, before_threshold, after_threshold = comparison
            comp_df = df.loc[(df[before] > before_threshold) | (df[after] > after_threshold), :]

            comp_out = pd.Series((comp_df[after] > after_threshold).astype(int).values, index=comp_df[seq_col])
            pos_weight = (len(comp_out) - comp_out.sum()) / comp_out.sum()
            comparison_cdr_to_output.append(comp_out.to_dict())
            self.comparison_pos_weights[i_comparison] = pos_weight
            print(f'comparison: {comparison}, {len(comp_out)} seqs, '
                  f'{comp_out.mean() * 100:0.1f}% positive, {pos_weight:0.4f} pos_weight')

        # keep only sequences with all output information
        all_cdrs = set.intersection(*(set(d.keys()) for d in comparison_cdr_to_output))
        df = df[df[seq_col].isin(all_cdrs)]
        self.all_cdr_seqs = df[seq_col].values
        print(f'total seqs after intersection: {len(self.all_cdr_seqs)}')

        # split data into train-test
        with temp_seed(self.split_seed):
            indices = np.random.permutation(len(self.all_cdr_seqs))
            partition = math.ceil(len(indices) * self.train_test_split)
            training_idx, test_idx = indices[:partition], indices[partition:]
            self.cdr_seqs_train, self.cdr_seqs_test = self.all_cdr_seqs[training_idx], self.all_cdr_seqs[test_idx]
            print(f'train-test split: {self.train_test_split}')
            print(f'num train, test seqs: {len(self.cdr_seqs_train)}, {len(self.cdr_seqs_test)}')

        # make table of output values
        self.cdr_to_output = {}
        for cdr in df[seq_col]:
            output = []
            for d in comparison_cdr_to_output:
                output.append(d.get(cdr, 0))
            self.cdr_to_output[cdr] = output

        df = df.set_index(seq_col)
        self.cdr_to_heavy = df[heavy_col].to_dict()
        self.cdr_to_light = df[light_col].to_dict()

    @property
    def n_eff(self):
        return len(self.cdr_seqs)

    @property
    def cdr_seqs(self):
        if self._training:
            return self.cdr_seqs_train
        else:
            return self.cdr_seqs_test

    def __getitem__(self, index):
        if self.unlimited_epoch:
            indices = np.random.randint(0, self.n_eff, self.batch_size)
        else:
            first_index = index * self.batch_size
            last_index = min((index+1) * self.batch_size, self.n_eff)
            indices = np.arange(first_index, last_index)

        seqs = self.cdr_seqs[indices].tolist()
        label_arr = torch.zeros(len(indices), len(self.comparisons))
        for i, seq in enumerate(seqs):
            for j, output in enumerate(self.cdr_to_output[seq]):
                label_arr[i, j] = output

        if len(seqs) == 0:
            return None
        vls = [self.cdr_to_light[cdr] for cdr in seqs]
        vhs = [self.cdr_to_heavy[cdr] for cdr in seqs]
        batch = self.sequences_to_onehot(seqs, vls=vls, vhs=vhs)
        batch['label'] = label_arr
        return batch


class VHAntibodyDataset(AntibodySequenceDataset):
    """Abstract antibody dataset"""
    IPI_VH_SEQS = ['IGHV1-46', 'IGHV1-69', 'IGHV3-7', 'IGHV3-15', 'IGHV4-39', 'IGHV5-51']  # TODO IGHV1-69D?
    LABELED = False

    def __init__(
            self,
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='encoder',
            include_vh=False,
            vh_set_name='IPI',
    ):
        super(VHAntibodyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
            include_vl=False,
            include_vh=include_vh,
        )
        self.vh_set_name = vh_set_name

        self._n_eff = 1
        if self.vh_set_name == 'IPI':
            self.vh_list = self.IPI_VH_SEQS.copy()
        else:
            self.vh_list = None

    @property
    def input_dim(self):
        input_dim = len(self.alphabet)
        if self.include_vh:
            input_dim += len(self.heavy_to_idx)
        return input_dim

    @property
    def params(self):
        params = super(VHAntibodyDataset, self).params
        params.pop('vl_seqs', None)
        params.pop('include_vl', None)
        params.update({
            "vh_set_name": self.vh_set_name,
            "vh_seqs": self.vh_list,
        })
        return params

    @params.setter
    def params(self, d):
        d.pop('vl_seqs', None)
        d.pop('include_vl', None)
        AntibodySequenceDataset.params.__set__(self, d)
        if 'vh_set_name' in d:
            self.vh_set_name = d['vh_set_name']
            if self.vh_set_name == 'IPI':
                self.vh_list = self.IPI_VH_SEQS.copy()
            else:
                self.vh_list = None
        if 'vh_seqs' in d:
            self.vh_list = d['vh_seqs']

    @property
    def n_eff(self):
        """Number of clusters across all VH genes"""
        return self._n_eff

    def __getitem__(self, index):
        raise NotImplementedError


class VHAntibodyFastaDataset(VHAntibodyDataset):
    """Antibody dataset with VH sequences.
    fasta: >seq(:.+)*:VH_gene
    """

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
            include_vh=False,
            vh_set_name='IPI',
    ):
        super(VHAntibodyFastaDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
            include_vh=include_vh,
            vh_set_name=vh_set_name,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.names = None
        self.vh_genes = None
        self.sequences = None

        self.load_data()

    def load_data(self):
        filename = os.path.join(self.working_dir, self.dataset)
        names_list = []
        vh_genes_list = []
        sequence_list = []

        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue

                names_list.append(title)
                vh_genes_list.append(title.split(':')[-1])
                sequence_list.append(seq)

        self.names = np.array(names_list)
        self.vh_genes = np.array(vh_genes_list)
        self.sequences = np.array(sequence_list)

        print("Number of sequences:", self.n_eff)

    @property
    def n_eff(self):
        return len(self.sequences)  # not a true n_eff

    def __getitem__(self, index):
        """
        :param index: batch index; ignored if unlimited_epoch
        :return: batch of size self.batch_size
        """

        if self.unlimited_epoch:
            indices = np.random.randint(0, self.n_eff, self.batch_size)
        else:
            first_index = index * self.batch_size
            last_index = min((index + 1) * self.batch_size, self.n_eff)
            indices = np.arange(first_index, last_index)

        seqs = self.sequences[indices]
        vhs = self.vh_genes[indices]
        batch = self.sequences_to_onehot(seqs, vhs=vhs)
        batch['names'] = self.names[indices]
        batch['sequences'] = [seq for seq, vh in seqs]
        return batch


class VHClusteredAntibodyDataset(VHAntibodyDataset):
    """Double-weighted antibody dataset.
    fasta: >seq:clu1:clu2
    clu1: VH gene
    clu2: cluster id
    """

    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            output_types='decoder',
            include_vh=False,
            vh_set_name='IPI',
    ):
        super(VHClusteredAntibodyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
            output_types=output_types,
            include_vh=include_vh,
            vh_set_name=vh_set_name,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.clu1_to_clu2s = {}
        self.clu1_to_clu2_to_seqs = {}

        self.load_data()

    @property
    def clu1_list(self):
        return self.vh_list

    def load_data(self):
        filename = self.working_dir + '/datasets/' + self.dataset
        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                name, clu1, clu2 = title.split(':')
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue

                if clu1 in self.clu1_to_clu2_to_seqs:
                    if clu2 in self.clu1_to_clu2_to_seqs[clu1]:
                        self.clu1_to_clu2_to_seqs[clu1][clu2].append(seq)
                    else:
                        self.clu1_to_clu2s[clu1].append(clu2)
                        self.clu1_to_clu2_to_seqs[clu1][clu2] = [seq]
                else:
                    self.clu1_to_clu2s[clu1] = [clu2]
                    self.clu1_to_clu2_to_seqs[clu1] = {clu2: [seq]}

        if self.clu1_list is None:
            self.vh_list = list(self.clu1_to_clu2_to_seqs.keys())
        self._n_eff = sum(len(clu2s) for clu2s in self.clu1_to_clu2s.values())
        print("Num VH genes:", len(self.clu1_list))
        print("N_eff:", self.n_eff)

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        seqs = []
        vhs = []
        for i in range(self.batch_size):
            # Pick a VH gene
            clu1_idx = np.random.randint(0, len(self.clu1_list))
            clu1 = self.clu1_list[clu1_idx]

            # Then pick a cluster for that VH gene
            clu2_list = self.clu1_to_clu2s[clu1]
            clu2_idx = np.random.randint(0, len(clu2_list))
            clu2 = clu2_list[clu2_idx]

            # Then pick a random sequence from the  cluster
            clu_seqs = self.clu1_to_clu2_to_seqs[clu1][clu2]
            seq_idx = np.random.randint(0, len(clu_seqs))
            seqs.append(clu_seqs[seq_idx])
            vhs.append(clu1)

        batch = self.sequences_to_onehot(seqs, vhs=vhs)
        return batch
