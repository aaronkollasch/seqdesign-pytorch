import os
import glob
import warnings
import math

import numpy as np
import torch
import torch.utils.data as data
from Bio.SeqIO.FastaIO import SimpleFastaParser

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


class SequenceDataset(GeneratorDataset):
    """Abstract sequence dataset"""
    def __init__(
            self,
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
    ):
        super(SequenceDataset, self).__init__(batch_size=batch_size, unlimited_epoch=unlimited_epoch)

        self.alphabet_type = alphabet_type
        self.reverse = reverse
        self.matching = matching

        # Load up the alphabet type to use, whether that be DNA, RNA, or protein
        if self.alphabet_type == 'protein':
            self.alphabet = PROTEIN_ALPHABET
            self.reorder_alphabet = PROTEIN_REORDERED_ALPHABET
        elif self.alphabet_type == 'RNA':
            self.alphabet = RNA_ALPHABET
            self.reorder_alphabet = RNA_ALPHABET
        elif self.alphabet_type == 'DNA':
            self.alphabet = DNA_ALPHABET
            self.reorder_alphabet = DNA_ALPHABET

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        self.idx_to_aa = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i
            self.idx_to_aa[i] = aa

    @property
    def params(self):
        params = super(SequenceDataset, self).params
        params.update({
            "alphabet_type": self.alphabet_type,
            "reverse": self.reverse,
            "matching": self.matching
        })
        return params

    @params.setter
    def params(self, d):
        GeneratorDataset.params.__set__(self, d)
        if 'alphabet_type' in d and d['alphabet_type'] != self.alphabet_type:
            warnings.warn(f"Cannot change alphabet type from {d['alphabet_type']} to {self.alphabet_type}")
        if 'reverse' in d:
            self.reverse = d['reverse']
        if 'matching' in d:
            self.matching = d['matching']

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
        num_seqs = len(sequences)
        max_seq_len = max([len(seq) for seq in sequences]) + 1
        prot_decoder_output = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))
        prot_decoder_input = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))

        if matching:
            prot_decoder_output_r = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))
            prot_decoder_input_r = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))

        prot_mask_decoder = torch.zeros((num_seqs, 1, 1, max_seq_len))

        for i, sequence in enumerate(sequences):
            if reverse:
                sequence = sequence[::-1]

            decoder_input_seq = '*' + sequence
            decoder_output_seq = sequence + '*'

            if matching:
                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*' + sequence_r
                decoder_output_seq_r = sequence_r + '*'

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, self.aa_dict[decoder_input_seq[j]], 0, j] = 1
                prot_decoder_output[i, self.aa_dict[decoder_output_seq[j]], 0, j] = 1
                prot_mask_decoder[i, 0, 0, j] = 1

                if matching:
                    prot_decoder_input_r[i, self.aa_dict[decoder_input_seq_r[j]], 0, j] = 1
                    prot_decoder_output_r[i, self.aa_dict[decoder_output_seq_r[j]], 0, j] = 1

        if matching:
            return {
                'prot_decoder_input': prot_decoder_input,
                'prot_decoder_output': prot_decoder_output,
                'prot_mask_decoder': prot_mask_decoder,
                'prot_decoder_input_r': prot_decoder_input_r,
                'prot_decoder_output_r': prot_decoder_output_r
            }
        else:
            return {
                'prot_decoder_input': prot_decoder_input,
                'prot_decoder_output': prot_decoder_output,
                'prot_mask_decoder': prot_mask_decoder
            }


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
    ):
        super(FastaDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching
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

        self.names = np.array(names_list)
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
    ):
        super(SingleFamilyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching
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

        self.seq_len = 0
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
        self.seq_len = max_seq_len
        self.num_families = len(self.family_name_list)
        self.max_family_size = max_family_size

        print("Number of families:", self.num_families)
        print("Neff:", np.sum(weight_list))
        print("Max family size:", max_family_size)

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


class DoubleWeightedNanobodyDataset(SequenceDataset):
    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
    ):
        super(DoubleWeightedNanobodyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.name_to_sequence = {}
        self.clu1_to_clu2_to_seq_names = {}
        self.clu1_to_clu2_to_clu_size = {}
        self.clu1_list = []

        self.load_data()

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

                self.name_to_sequence[name] = seq
                if clu1 in self.clu1_to_clu2_to_seq_names:
                    if clu2 in self.clu1_to_clu2_to_seq_names[clu1]:
                        self.clu1_to_clu2_to_seq_names[clu1][clu2].append(name)
                        self.clu1_to_clu2_to_clu_size[clu1][clu2] += 1
                    else:
                        self.clu1_to_clu2_to_seq_names[clu1][clu2] = [name]
                        self.clu1_to_clu2_to_clu_size[clu1][clu2] = 1
                else:
                    self.clu1_to_clu2_to_seq_names[clu1] = {clu2: [name]}
                    self.clu1_to_clu2_to_clu_size[clu1] = {clu2: 1}

        self.clu1_list = list(self.clu1_to_clu2_to_seq_names.keys())
        print("Num clusters:", len(self.clu1_list))

    @property
    def n_eff(self):
        return len(self.clu1_list)

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        seqs = []
        for i in range(self.batch_size):
            # Pick a cluster id80
            clu1_idx = np.random.randint(0, len(self.clu1_list))
            clu1 = self.clu1_list[clu1_idx]

            # Then pick a cluster id90 from the cluster id80s
            clu2 = np.random.choice(list(self.clu1_to_clu2_to_seq_names[clu1].keys()))

            # Then pick a random sequence all in those clusters
            seq_name = np.random.choice(self.clu1_to_clu2_to_seq_names[clu1][clu2])

            # then grab the associated sequence
            seqs.append(self.name_to_sequence[seq_name])

        batch = self.sequences_to_onehot(seqs)
        return batch


class VHAntibodyDataset(SequenceDataset):
    """Abstract antibody dataset"""

    IPI_VH_SEQS = ['IGHV1-46', 'IGHV1-69', 'IGHV3-7', 'IGHV3-15', 'IGHV4-39', 'IGHV5-51']  # TODO IGHV1-69D?

    def __init__(
            self,
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            include_vh=False,
            vh_set_name='IPI',
    ):
        super(VHAntibodyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching
        )
        self.include_vh = include_vh
        self.vh_set_name = vh_set_name

        self._n_eff = 1
        if self.vh_set_name == 'IPI':
            self.vh_list = self.IPI_VH_SEQS.copy()
        else:
            self.vh_list = None

    @property
    def heavy_to_idx(self):
        if self.vh_list is None:
            raise RuntimeError("VH list not loaded.")
        else:
            return {vh: i for i, vh in enumerate(self.vh_list)}

    @property
    def input_dim(self):
        input_dim = len(self.alphabet)
        if self.include_vh:
            input_dim += len(self.heavy_to_idx)

    @property
    def params(self):
        params = super(VHAntibodyDataset, self).params
        params.update({
            "include_vh": self.include_vh,
            "vh_set_name": self.vh_set_name,
            "vh_seqs": self.vh_list,
        })
        return params

    @params.setter
    def params(self, d):
        SequenceDataset.params.__set__(self, d)
        if 'include_vh' in d:
            self.include_vh = d['include_vh']
        if 'vh_set_name' in d:
            self.vh_set_name = d['vh_set_name']
        if 'vh_seqs' in d:
            self.vh_list = d['vh_seqs']

    @property
    def n_eff(self):
        """Number of clusters across all VH genes"""
        return self._n_eff

    def __getitem__(self, index):
        raise NotImplementedError

    def sequences_to_onehot(self, sequences, reverse=None, matching=None, include_vh=None):
        """

        :param sequences: list(tuple(seq, VH gene))
        :param reverse:
        :param matching:
        :param include_vh:
        :return:
        """
        reverse = self.reverse if reverse is None else reverse
        matching = self.matching if matching is None else matching
        include_vh = self.include_vh if include_vh is None else include_vh
        num_seqs = len(sequences)
        max_seq_len = max([len(seq) for seq, vh in sequences]) + 1

        prot_decoder_output = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))
        prot_decoder_input = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))
        if matching:
            prot_decoder_output_r = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))
            prot_decoder_input_r = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))
        prot_mask_decoder = torch.zeros((num_seqs, 1, 1, max_seq_len))
        heavy_arr = torch.zeros(num_seqs, len(self.heavy_to_idx), 1, max_seq_len)

        for i, (sequence, vh) in enumerate(sequences):
            if reverse:
                sequence = sequence[::-1]

            decoder_input_seq = '*' + sequence
            decoder_output_seq = sequence + '*'

            if matching:
                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*' + sequence_r
                decoder_output_seq_r = sequence_r + '*'

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, self.aa_dict[decoder_input_seq[j]], 0, j] = 1
                prot_decoder_output[i, self.aa_dict[decoder_output_seq[j]], 0, j] = 1
                prot_mask_decoder[i, 0, 0, j] = 1
                heavy_arr[i, self.heavy_to_idx[vh], 0, j] = 1

                if matching:
                    prot_decoder_input_r[i, self.aa_dict[decoder_input_seq_r[j]], 0, j] = 1
                    prot_decoder_output_r[i, self.aa_dict[decoder_output_seq_r[j]], 0, j] = 1

        if include_vh:
            prot_decoder_input = torch.cat([prot_decoder_input, heavy_arr], dim=1)
            if matching:
                prot_decoder_input_r = torch.cat([prot_decoder_input_r, heavy_arr], dim=1)

        if matching:
            return {
                'prot_decoder_input': prot_decoder_input,
                'prot_decoder_output': prot_decoder_output,
                'prot_mask_decoder': prot_mask_decoder,
                'prot_decoder_input_r': prot_decoder_input_r,
                'prot_decoder_output_r': prot_decoder_output_r
            }
        else:
            return {
                'prot_decoder_input': prot_decoder_input,
                'prot_decoder_output': prot_decoder_output,
                'prot_mask_decoder': prot_mask_decoder
            }


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
            include_vh=False,
    ):
        super(VHAntibodyFastaDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            include_vh=include_vh,
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

        seqs = list(zip(self.sequences[indices], self.vh_genes[indices]))
        batch = self.sequences_to_onehot(seqs)
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
            include_vh=False,
    ):
        super(VHClusteredAntibodyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            include_vh=include_vh,
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
            seqs.append((clu_seqs[seq_idx], clu1))

        batch = self.sequences_to_onehot(seqs)
        return batch
