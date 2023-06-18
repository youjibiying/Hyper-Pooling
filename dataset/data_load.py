import scipy.sparse
from scipy.sparse import coo_matrix
import numpy as np
import scipy.linalg
from scipy.special import softmax
from sklearn.preprocessing import scale, MinMaxScaler
import os
import random
import pandas as pd
import time
import copy
import pickle
import gzip, glob
import torch
from .s2vgraph import S2VGraph
from .arxiv_dataset import ArXiv


def readLinesStrip(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines


def onek_encoding_unk(value: int, choices):
    """
    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    if min(choices) < 0:
        index = value
    else:
        index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


class Protein:
    def __init__(self, data_path=None, args=None):
        self.args = args
        self.path = data_path if data_path is not None else args.data_path
        self.protein_path = 'protein'
        self.gpcr_path = os.path.join(self.path, self.protein_path, 'gpcr')
        self.kinase_path = os.path.join(self.path, self.protein_path, 'kinase')
        self.amino_acid = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.n_class = 2
        self.fea_dim = 21
        self.g_list = []

    def get_gpcr_kinase(self):
        gpcr_kinase_file = os.path.join(self.gpcr_path + '_kinase.pkl.gz')
        if os.path.exists(gpcr_kinase_file):
            with gzip.open(gpcr_kinase_file) as fp:
                self.g_list = pickle.load(fp)
        else:
            print('Loading protein dataset...')
            gpcr_list = self.get_protein_file(self.gpcr_path, label=1)
            kinase_list = self.get_protein_file(self.kinase_path, label=0)
            self.g_list = gpcr_list + kinase_list
            try:
                with gzip.open(gpcr_kinase_file, 'wb') as f:
                    pickle.dump(self.g_list, f)
            except:
                print("g_list fail to save")
        if len(self.g_list) == 0:
            raise ValueError('protein dataset is null, may be path is wrong!')
        return self.g_list

    def get_protein_file(self, path, label):
        g_list = []
        for root, dirs, files in os.walk(path):
            for fname in files:
                seq, contactmap = self.getProtein(path, fname)
                contact = np.array(contactmap, dtype=np.float)

                row, col, data = contact[:, 0].astype(np.int32), contact[:, 1].astype(np.int32), contact[:, -1]
                size = len(seq)
                coo = coo_matrix((data, (row, col)), shape=(size, max(col) + 1))
                node_features = self.amino_acid_features(seq)
                g_list.append(S2VGraph(coo, label, node_features=node_features))
        return g_list

    def amino_acid_features(self, seq: str, idx=None):  #
        """
        Builds a feature vector for an amino acid.
        :param seq: An amino acid sequence of protein.
        :param idx:
        :return: A list containing the protein node features.
        """

        one_hot_list = list(range(len(self.amino_acid)))
        node_features = []
        for amino in seq:
            f_amino = onek_encoding_unk(self.amino_acid.index(amino), one_hot_list)
            node_features.append(f_amino)
        # node_features=torch.tensor(node_features)
        # if self.args.B_features:
        #     f_amino += [1 if i in self.dit_residue[amino] else 0 for i in range(8)] + [
        #         self.mol_weight[amino]]  # one hot encoding
        # if self.args.C_features:
        #     f_amino += self.secondary_structure[idx]
        # if self.args.D_features:
        #     f_amino += [self.B_factor[idx]]
        # if self.args.E_features:
        #     f_amino += [int(self.conservation_score[idx])]

        return node_features

    def getProtein(self, path, contactMapName, contactMap=True):
        proteins = open(path + "/" + contactMapName).readlines()
        proteins = readLinesStrip(proteins)
        seq = proteins[1]
        if (contactMap):
            contactMap = []
            for i in range(2, len(proteins)):
                line = [float(i) for i in proteins[i].split()]
                contactMap.append(line)
            return seq, contactMap
        else:
            return seq


def load_data(data_path, dataset: str, args=None):
    if dataset.lower() == 'proteins':
        protein = Protein(data_path=data_path)
        graphs, num_classes, fea_dim = protein.get_gpcr_kinase(), protein.n_class, protein.fea_dim
    elif dataset.lower() == 'arxiv':
        arxiv_dataset = ArXiv(data_path=data_path)
        graphs, num_classes, fea_dim = [
            arxiv_dataset.get_graphs(),
            arxiv_dataset.num_classes,
            arxiv_dataset.feature_dim]
    else:
        raise ValueError(f" {dataset} dataset don't exist")
    return graphs, num_classes, fea_dim


if __name__ == '__main__':
    data_path = './'
    g, _, _ = load_data(data_path=data_path, dataset='Proteins')
    print("size of dataset:", len(g))
