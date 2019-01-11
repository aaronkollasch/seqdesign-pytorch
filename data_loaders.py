import os
import glob
import time

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
    def __init__(
            self,
            batch_size=32,
            unlimited_epoch=True,
    ):
        self.batch_size = batch_size
        self.unlimited_epoch = unlimited_epoch

    @property
    def n_eff(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.unlimited_epoch:
            return 2 ** 62
        else:
            return int(np.ceil(self.n_eff / self.batch_size))

    @staticmethod
    def collate_fn(batch):
        return batch[0]


class GeneratorDataLoader(data.DataLoader):
    def __init__(self, dataset: GeneratorDataset, **kwargs):
        kwargs.update(dict(
            batch_size=1, shuffle=False, sampler=None, batch_sampler=None, collate_fn=dataset.collate_fn,
        ))
        super(GeneratorDataLoader, self).__init__(
            dataset,
            **kwargs)


class SequenceDataset(GeneratorDataset):
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
    def n_eff(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def sequences_to_onehot(self, sequences, reverse=False, matching=False):
        """

        :param sequences: list of strings
        :param reverse: reverse the sequences
        :param matching: output forward and reverse sequences
        :return: dictionary of strings
        """
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
        batch = self.sequences_to_onehot(seqs, self.reverse, self.matching)
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

        # self.protein_mutation_names = []
        # self.protein_names_to_uppercase_idx = {}
        # self.protein_names_to_one_hot_seqs_encoder = {}
        # self.mut_protein_names_to_seqs_encoder = {}
        # self.protein_names_to_one_hot_seqs_decoder_input_f = {}
        # self.protein_names_to_one_hot_seqs_decoder_output_f = {}
        # self.protein_names_to_one_hot_seqs_decoder_input_r = {}
        # self.protein_names_to_one_hot_seqs_decoder_output_r = {}
        # self.protein_names_to_one_hot_seqs_encoder_mask = {}
        # self.protein_names_to_one_hot_seqs_decoder_mask = {}
        # self.protein_names_to_measurement_list = {}
        # self.protein_names_to_mutation_list = {}

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

        # # Read in the mutation data so we can predict it later
        # self.protein_mutation_names = []
        # self.protein_names_to_uppercase_idx = {}
        # self.protein_names_to_one_hot_seqs_encoder = {}
        # self.mut_protein_names_to_seqs_encoder = {}
        # self.protein_names_to_one_hot_seqs_decoder_input_f = {}
        # self.protein_names_to_one_hot_seqs_decoder_output_f = {}
        # self.protein_names_to_one_hot_seqs_decoder_input_r = {}
        # self.protein_names_to_one_hot_seqs_decoder_output_r = {}
        # self.protein_names_to_one_hot_seqs_encoder_mask = {}
        # self.protein_names_to_one_hot_seqs_decoder_mask = {}
        # self.protein_names_to_measurement_list = {}
        # self.protein_names_to_mutation_list = {}
        #
        # for filename in glob.glob(self.working_dir+'/datasets/mutation_data/'+self.dataset+'*.csv'):
        #     sequence_list = []
        #     mutation_list = []
        #     uppercase_list = []
        #     measurement_list = []
        #
        #     family_name_list = filename.split('/')[-1].split('_')
        #     family_name = family_name_list[0]+'_'+family_name_list[1]
        #
        #     mutation_counter = 0
        #     with open(filename, 'r') as f_in:
        #         for i, line in enumerate(f_in):
        #             line = line.rstrip()
        #             line_list = line.split(',')
        #             if i != 0:
        #                 mutation, is_upper, measurement, sequence = line_list
        #
        #                 measurement = float(measurement)
        #                 if np.isfinite(measurement):
        #                     if is_upper == 'True':
        #                         uppercase_list.append(mutation_counter)
        #                     mutation_list.append(mutation)
        #                     measurement_list.append(float(measurement))
        #
        #                     sequence_list.append(sequence)
        #                     seq_len_mutations = len(sequence)
        #                     mutation_counter += 1
        #
        #     self.protein_mutation_names.append(family_name)
        #     self.protein_names_to_uppercase_idx[family_name] = np.asarray(uppercase_list)
        #     self.protein_names_to_measurement_list[family_name] = np.asarray(measurement_list)
        #     self.protein_names_to_mutation_list[family_name] = mutation_list
        #     self.mut_protein_names_to_seqs_encoder[family_name] = sequence_list
        #
        #     prot_encoder = np.zeros((len(measurement_list), 1, seq_len_mutations, len(self.alphabet)))
        #     prot_mask_encoder = np.zeros((len(measurement_list), 1, seq_len_mutations, 1))
        #     prot_decoder_output_f = np.zeros((len(measurement_list), 1, seq_len_mutations+1, len(self.alphabet)))
        #     prot_decoder_input_f = np.zeros((len(measurement_list), 1, seq_len_mutations+1, len(self.alphabet)))
        #     prot_decoder_output_r = np.zeros((len(measurement_list), 1, seq_len_mutations+1, len(self.alphabet)))
        #     prot_decoder_input_r = np.zeros((len(measurement_list), 1, seq_len_mutations+1, len(self.alphabet)))
        #     prot_mask_decoder = np.zeros((len(measurement_list), 1, seq_len_mutations+1, 1))
        #
        #     for j,sequence in enumerate(sequence_list):
        #         for k,letter in enumerate(sequence):
        #             if letter in self.aa_dict:
        #                 l = self.aa_dict[letter]
        #                 prot_encoder[j,0,k,l] = 1.0
        #                 prot_mask_encoder[j,0,k,0] = 1.0
        #
        #     for j,sequence in enumerate(decoder_output_sequence_list):
        #         for k,letter in enumerate(sequence):
        #             if letter in self.aa_dict:
        #                 l = self.aa_dict[letter]
        #                 prot_decoder_output_f[j,0,k,l] = 1.0
        #                 prot_mask_decoder[j,0,k,0] = 1.0
        #
        #     for j,sequence in enumerate(decoder_input_sequence_list):
        #         for k,letter in enumerate(sequence):
        #             if letter in self.aa_dict:
        #                 l = self.aa_dict[letter]
        #                 prot_decoder_input_f[j,0,k,l] = 1.0
        #
        #     for j,sequence in enumerate(encoder_sequence_list):
        #         sequence_r = "*"+sequence[::-1]
        #         for k,letter in enumerate(sequence_r):
        #             if letter in self.aa_dict:
        #                 l = self.aa_dict[letter]
        #                 prot_decoder_input_r[j,0,k,l] = 1.0
        #
        #     for j,sequence in enumerate(encoder_sequence_list):
        #         sequence_r = sequence[::-1]+'*'
        #         for k,letter in enumerate(sequence_r):
        #             if letter in self.aa_dict:
        #                 l = self.aa_dict[letter]
        #                 prot_decoder_output_r[j,0,k,l] = 1.0
        #
        #     self.protein_names_to_one_hot_seqs_encoder[family_name] = prot_encoder
        #     self.protein_names_to_one_hot_seqs_encoder_mask[family_name] = prot_mask_encoder
        #     self.protein_names_to_one_hot_seqs_decoder_input_f[family_name] = prot_decoder_input_f
        #     self.protein_names_to_one_hot_seqs_decoder_output_f[family_name] = prot_decoder_output_f
        #     self.protein_names_to_one_hot_seqs_decoder_input_r[family_name] = prot_decoder_input_r
        #     self.protein_names_to_one_hot_seqs_decoder_output_r[family_name] = prot_decoder_output_r
        #     self.protein_names_to_one_hot_seqs_decoder_mask[family_name] = prot_mask_decoder

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

        batch = self.sequences_to_onehot(seqs, self.reverse, self.matching)
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

        batch = self.sequences_to_onehot(seqs, self.reverse, self.matching)
        return batch


class DataHelperDoubleWeightingNanobody:
    def __init__(self, alignment_file='', focus_seq_name='',
                 mutation_file='', calc_weights=True, working_dir='.', alphabet_type='protein'):
        np.random.seed(42)
        self.alignment_file = alignment_file
        self.focus_seq_name = focus_seq_name
        self.mutation_file = mutation_file
        self.working_dir = working_dir
        self.calc_weights = calc_weights
        self.alphabet_type = alphabet_type

        self.aa_dict = {}
        self.idx_to_aa = {}
        self.num_to_aa = {}
        self.name_to_sequence = {}
        self.clu80_to_clu90_to_seq_names = {}
        self.clu80_to_clu90_to_clu_size = {}
        self.cluster_id80_list = []

        # Load up the alphabet type to use, whether that be DNA, RNA, or protein
        self.alphabet, self.reorder_alphabet = get_alphabet(alphabet_type)

        # Make a dictionary that goes from aa to a number for one-hot
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i
            self.idx_to_aa[i] = aa

        # Do the inverse as well
        self.num_to_aa = {i: aa for aa, i in self.aa_dict.items()}

        # then generate the experimental data
        self.gen_alignment_mut_data()

    def gen_alignment_mut_data(self):
        ix = np.array([self.alphabet.find(s) for s in self.reorder_alphabet])

        with open(self.working_dir+'/datasets/'+self.alignment_file, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                name, clu80, clu90 = title.split(':')
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue

                self.name_to_sequence[name] = seq
                if clu80 in self.clu80_to_clu90_to_seq_names:
                    if clu90 in self.clu80_to_clu90_to_seq_names[clu80]:
                        self.clu80_to_clu90_to_seq_names[clu80][clu90].append(name)
                        self.clu80_to_clu90_to_clu_size[clu80][clu90] += 1
                    else:
                        self.clu80_to_clu90_to_seq_names[clu80][clu90] = [name]
                        self.clu80_to_clu90_to_clu_size[clu80][clu90] = 1
                else:
                    self.clu80_to_clu90_to_seq_names[clu80] = {clu90: [name]}
                    self.clu80_to_clu90_to_clu_size[clu80] = {clu90: 1}

        self.cluster_id80_list = list(self.clu80_to_clu90_to_seq_names.keys())
        print("Num clusters:", len(self.cluster_id80_list))

    def generate_one_family_minibatch_data(
            self, minibatch_size, use_embedding=True, reverse=False, matching=False, top_k=False):

        start = time.time()
        n_eff = len(self.cluster_id80_list)
        cluster_name_list = np.random.choice(self.cluster_id80_list, minibatch_size).tolist()

        minibatch_max_seq_len = 0
        sequence_list = []
        for i, cluster_id80 in enumerate(cluster_name_list):

            # First pick a cluster id90 from the cluster id80s
            cluster_id90 = np.random.choice(list(self.clu80_to_clu90_to_seq_names[cluster_id80].keys()), 1)[0]

            # Then pick a random sequence all in those clusters
            if self.clu80_to_clu90_to_clu_size[cluster_id80][cluster_id90] > 1:
                seq_name = np.random.choice(self.clu80_to_clu90_to_seq_names[cluster_id80][cluster_id90], 1)[0]
            else:
                seq_name = self.clu80_to_clu90_to_seq_names[cluster_id80][cluster_id90][0]

            # then grab the associated sequence
            seq = self.name_to_sequence[seq_name]

            sequence_list.append(seq)

            seq_len = len(seq)
            if seq_len > minibatch_max_seq_len:
                minibatch_max_seq_len = seq_len

        # Add 1 to compensate for the start and end character
        minibatch_max_seq_len += 1

        out = tuple([*self.seq_list_to_minibatch(
            sequence_list, minibatch_size, minibatch_max_seq_len, reverse, matching), n_eff])
        return out

    def seq_list_to_minibatch(
            self, sequence_list, minibatch_size, minibatch_max_seq_len, reverse=False, matching=False):
        prot_decoder_output = torch.zeros((minibatch_size, len(self.alphabet), 1, minibatch_max_seq_len))
        prot_decoder_input = torch.zeros((minibatch_size, len(self.alphabet), 1, minibatch_max_seq_len))

        if matching:
            prot_decoder_output_r = torch.zeros((minibatch_size, len(self.alphabet), 1, minibatch_max_seq_len))
            prot_decoder_input_r = torch.zeros((minibatch_size, len(self.alphabet), 1, minibatch_max_seq_len))

        prot_mask_decoder = torch.zeros((minibatch_size, 1, 1, minibatch_max_seq_len))

        for i, sequence in enumerate(sequence_list):

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
            return (
                prot_decoder_input, prot_decoder_output, prot_mask_decoder,
                prot_decoder_input_r, prot_decoder_output_r
            )
        else:
            return prot_decoder_input, prot_decoder_output, prot_mask_decoder

    def seq_list_to_one_hot(self, sequence_list):
        seq_arr = np.zeros((len(sequence_list), len(self.alphabet), 1, len(sequence_list[0])))
        for i, seq in enumerate(sequence_list):
            for j, aa in enumerate(seq):
                k = self.aa_dict[aa]
                seq_arr[i, k, 0, j] = 1.
        return seq_arr

