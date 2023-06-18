from typing import List
import numpy as np
from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist
import os


def To_pocket_amino(contactmap, features_table):
    pocket_index = np.where(features_table[:, 2] == 1)[0]
    features_table = features_table[pocket_index, :]
    contactmap = contactmap[pocket_index, :][:, pocket_index]
    seq = ''.join(features_table[:, 0])
    return seq, contactmap, features_table


def compute_pairwise_distances(protein_xyz, protein1_xyz):
    """Takes an input m x 3 and n x 3 np arrays of 3D coords of protein and ligand,
    respectively, and outputs an m x n np array of pairwise distances in Angstroms
    between protein and ligand atoms. entry (i,j) is dist between the i"th protein
    atom and the j"th ligand atom.
    """

    pairwise_distances = cdist(protein_xyz, protein1_xyz, metric='euclidean')
    return (pairwise_distances)


def two_seq_compare(new_contactmap_seq, contactmap_seq):
    n = 0
    for i, j in zip(contactmap_seq, new_contactmap_seq):
        n += 1
        if i != j:  # 当 两个序列不相同的时候,会导致new_contactMap中的edge features合不进contact map
            for ni in range(min(len(i), len(j))):
                if i[ni] != j[ni]:
                    break
            print('The first place of difference :')
            print(ni, i[:ni + 1])
            raise ValueError(
                f'the seq of contactMap is different from new_contactMap at number {n} \n the contact map seq is: "{i}" \n'
                f'the new contact map seq is: "{j}"')
    print('The number of protein is: ', len(contactmap_seq))


def load_mass(path='./mass.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
    dit = {}
    for i in lines:
        lit = i.split()
        dit[lit[0] + lit[1]] = int(lit[3])
    return dit


def add_name_seq(dist_map, protein_name, seq):
    dist_map = dist_map.tolist()
    dist_map.insert(0, seq)
    dist_map.insert(0, protein_name)
    return dist_map


def save_contactMap(savedir, dist):
    with open(savedir, 'w') as f:
        for i in range(len(dist)):
            jointsFrame = dist[i]  # 每行
            if i < 2:
                f.writelines(jointsFrame)
                f.write('\n')
                continue
            for Ji in jointsFrame:
                strNum = str(int(Ji))
                f.write(strNum)
                f.write(' ')
            f.write('\n')


def generate_new_contactmap(mass_dict: dict, dit: dict, protein_3Dpath: str = './', save_dir: str = './',
                            contactmap_coo_dict: dict = {},
                            d0: int = 3.8, generate_features_table_csv: bool = False, only_pocket_amino: bool = False):
    seqlist = []
    save_dir1 = save_dir + '/new_contactMap/'
    if not os.path.exists(save_dir1) and not generate_features_table_csv:
        os.makedirs(save_dir1)
    for root, dirs, files in os.walk(protein_3Dpath):  # 在checkpoint路径中访问所有的文件
        for fname in files:  # 遍历所有的文件
            if not fname.endswith('.pdb'):
                continue
            name = fname.split('.')[0].split('_')[0].lower()

            with open(os.path.join(protein_3Dpath, fname), 'r') as f:
                i_lines = f.readlines()

            # obtain the FASTA sequence & atom coordinates
            chain_id = None

            idx_resd_prev = None  # 记录上一次放进距离列表的蛋白质序号
            seq = ""
            mass_cal = []
            i_mass = []
            protein_set = [0]
            coordinate = []
            nums = 0
            CB = [0, 0, 0]
            resd_pre = None
            for num, i_line in enumerate(i_lines):
                # skip out-of-domain residues
                # if seq == 'PKNTCKLLVVADHRFYRYMGRGEESTTTNYLIELIDRVDDIYRNTAWDNAGFKGYGIQIEQIRILKSPQGEKHYNMAKSYPNEEKDAWDVKMLLEQFSFDIAEEASKVCLAHLFTYQDFDMGTLGLAYVGSPRANSHGGVCPKAYYSKNIYLNSGLTSTKNYGKTILTKEADLVTTHELGHNFGAEHDPDGLAECAPNEDQGGKYVMYPIAVSGDHENNKMFSQCSKQSIYKTIESKAQECFQR':
                #     print(i_line)
                #     if num==4612:
                #         print(i_line)
                if i_line.startswith('ENDMDL'):
                    break  # only read the first model
                if not i_line.startswith('ATOM'):
                    if num == len(i_lines) - 1:
                        if len(protein_set) > 1:
                            mass_xyz = [0, 0, 0]
                            for i in range(3):
                                for j in range(len(mass_cal)):
                                    mass_xyz[i] += (mass_cal[j][i] * i_mass[j] / sum(i_mass))
                            seq += dit[protein_set[-1].upper()]
                            coordinate.append(mass_xyz)
                            break
                        else:
                            seq += dit[resd_pre]
                            coordinate.append(CB)
                    continue
                if chain_id is None:
                    chain_id = i_line[21]
                elif i_line[21] != chain_id:
                    # if len(protein_set) > 1:
                    #     mass_xyz = [0, 0, 0]
                    #     for i in range(3):
                    #         for j in range(len(mass_cal)):
                    #             mass_xyz[i] += (mass_cal[j][i] * i_mass[j] / sum(i_mass))
                    #     seq += DIT[protein_set[-1].upper()]
                    #     coordinate.append(mass_xyz)
                    #     protein_set = [0]
                    continue

                # initialize name(s) & atom coordinates for one or more residues
                idx_resd_w_sfx = i_line[22:27].strip()

                # obtain name & atom coordinates for the current residue
                atom_name = i_line[12:16].strip()
                resd_name = i_line[17:20]

                cord_x = float(i_line[30:38])
                cord_y = float(i_line[38:46])
                cord_z = float(i_line[46:54])

                if idx_resd_prev != None and idx_resd_prev != idx_resd_w_sfx:
                    mass_xyz = [0, 0, 0]
                    if len(protein_set) == 1:
                        # print(name, resd_pre)
                        seq += dit[resd_pre]
                        coordinate.append(CB)
                    else:
                        for i in range(3):
                            for j in range(len(mass_cal)):
                                mass_xyz[i] += (mass_cal[j][i] * i_mass[j] / sum(i_mass))
                        # if len(protein_set)>2:
                        #     seq[]
                        # else:
                        seq += dit[protein_set[-1].upper()]
                        coordinate.append(mass_xyz)
                    i_mass = []
                    mass_cal = []
                    protein_set = [0]

                idx_resd_prev = idx_resd_w_sfx

                if atom_name + resd_name in mass_dict:
                    # if len(protein_set) == 1 or idx_resd_prev == idx_resd_w_sfx or idx_resd_prev == None:
                    mass_cal.append([cord_x, cord_y, cord_z])
                    i_mass.append(mass_dict[atom_name + resd_name])
                    protein_set.append(resd_name)

                if atom_name == 'CA' or atom_name == 'CB':
                    CB = [cord_x, cord_y, cord_z]
                    resd_pre = resd_name
                    # idx_resd_prev = idx_resd_w_sfx

                if num == len(i_lines) - 1:
                    if len(protein_set) > 1:
                        mass_xyz = [0, 0, 0]
                        for i in range(3):
                            for j in range(len(mass_cal)):
                                mass_xyz[i] += (mass_cal[j][i] * i_mass[j] / sum(i_mass))
                        seq += dit[protein_set[-1].upper()]
                        coordinate.append(mass_xyz)
                    else:
                        seq += dit[resd_pre]
                        coordinate.append(CB)

            seqlist.append(seq)
            if generate_features_table_csv:
                continue
            cords = np.array(coordinate, dtype=np.float32)
            dist = compute_pairwise_distances(cords, cords)
            dist = 1 / (1 + dist / d0)
            if only_pocket_amino:
                seq, dist, features = To_pocket_amino(dist, pd.read_csv(
                    os.path.join(save_dir, 'protein_features', name.lower() + '_features'), sep=' ',
                    header=None).to_numpy())
            contactmap_coo = contactmap_coo_dict[name]
            newcontactmap_coo = contactmap_coo.copy()
            for k, (i, j, _) in enumerate(contactmap_coo):
                newcontactmap_coo[k, 2] = dist[int(i), int(j)]
            dist = add_name_seq(dist_map=newcontactmap_coo, protein_name=name, seq=seq)
            save_contactMap(save_dir1 + name + "_full", dist)

    print("protein mass data processing successfully")
    return seqlist


def generate_contactmap(mass_dict: dict, dit: dict, protein_3Dpath: str = './', save_dir: str = './',
                        d0: int = 3.8, generate_features_table_csv: bool = False, only_pocket_amino: bool = False):
    def _parse_idx_w_sfx(idx_w_sfx):
        return (int(idx_w_sfx), ' ') \
            if idx_w_sfx[-1] in '0123456789' else (int(idx_w_sfx[:-1]), idx_w_sfx[-1])

    def _calc_nb_resds_diff(idx_resd_w_sfx_prev, idx_resd_w_sfx_curr):
        if idx_resd_w_sfx_prev is None:
            return 1
        idx_resd_prev, sfx_resd_prev = _parse_idx_w_sfx(idx_resd_w_sfx_prev)
        idx_resd_curr, sfx_resd_curr = _parse_idx_w_sfx(idx_resd_w_sfx_curr)
        nb_resds_diff = idx_resd_curr - idx_resd_prev
        if sfx_resd_curr != ' ':
            nb_resds_diff += 1
        return nb_resds_diff

    seqlist = []
    # save_dir1 = save_dir + '/contactMap/'
    # save_feature_dir = save_dir + '/protein_features'
    number = 0
    mu = 5.8
    k = 10
    w = 5
    contactmap_coo_dict = {}
    avg_edges_degree = []
    avg_nodes = []
    avg_edges = []
    # if not os.path.exists(save_dir1) and not generate_features_table_csv:
    #     os.makedirs(save_dir1)
    for root, dirs, files in os.walk(protein_3Dpath):  # 在checkpoint路径中访问所有的文件

        for fname in files:  # 遍历所有的文件
            if not fname.endswith('.pdb'):
                continue
            number += 1
            print(number, os.path.join(protein_3Dpath, fname))
            name = fname.split('.')[0].split('_')[0].lower()
            # if os.path.exists(os.path.join(protein_3Dpath, name + '_features.csv')) and not generate_features_table_csv:
            #     df = pd.read_csv(os.path.join(protein_3Dpath, name + '_features.csv'), header=0, sep=',')
            #     # df = df.fillna(0)
            #     features = np.array(df)
            # else:
            #     features = np.array([])
            with open(os.path.join(protein_3Dpath, fname), 'r') as f:
                i_lines = f.readlines()
            # obtain the FASTA sequence & atom coordinates
            temp = []
            resd_names = []
            chain_id = None
            cords_raw, masks_raw = [], []
            idx_resd_w_sfx_prev = None
            seq = ""

            coordinate = []
            f_idx = 0
            attn_list = []
            for num, i_line in enumerate(i_lines):
                # skip out-of-domain residues
                # if name.upper()=='PE2R3':
                #     print(i_line)
                # if seq == 'GAPPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSS':
                #     print(fname,i_line)
                #     if num==968:
                #         print(i_line)
                if i_line.startswith('ENDMDL'):
                    break  # only read the first model
                if not i_line.startswith('ATOM'):
                    continue
                if chain_id is None:
                    chain_id = i_line[21]
                elif i_line[21] != chain_id:
                    continue

                idx_resd_w_sfx = i_line[22:27].strip()
                idx_resd, sfx_resd = _parse_idx_w_sfx(idx_resd_w_sfx)

                # initialize name(s) & atom coordinates for one or more residues
                if idx_resd_w_sfx_prev is None or idx_resd_w_sfx_prev != idx_resd_w_sfx:
                    nb_resds_diff = _calc_nb_resds_diff(idx_resd_w_sfx_prev, idx_resd_w_sfx)
                    resd_names.extend(['XAA'] * nb_resds_diff)
                    cords_raw.extend([[0.0, 0.0, 0.0]] * nb_resds_diff)
                    masks_raw.extend([0] * nb_resds_diff)
                    idx_resd_w_sfx_prev = idx_resd_w_sfx

                # obtain name & atom coordinates for the current residue
                atom_name = i_line[12:16].strip()
                resd_name = i_line[17:20]
                if ((resd_name == 'GLY' or resd_name == 'GLZ') and atom_name != 'CA') or (
                        resd_name != 'GLZ' and resd_name != 'GLY' and atom_name != 'CB'):
                    continue

                if i_line[16] == 'C' or i_line[16] == 'B':
                    continue
                cord_x = float(i_line[30:38])
                cord_y = float(i_line[38:46])
                cord_z = float(i_line[46:54])

                temp.append([atom_name, resd_name])
                if resd_name.upper() not in dit:
                    print([name, resd_name.upper()])
                    continue
                seq += dit[resd_name.upper()]
                coordinate.append([cord_x, cord_y, cord_z])
                # if features.size > 0:
                #     while int(idx_resd_w_sfx) != int(features[f_idx][0]):
                #         print(int(idx_resd_w_sfx), features[f_idx][0])
                #         features = np.delete(features, f_idx, 0)
                #     f_idx += 1
                # else:
                #     attn_list.append([idx_resd_w_sfx, seq[-1]])

            seqlist.append(seq)
            # if features.size > 0:
            #     if not os.path.exists(save_feature_dir):
            #         os.makedirs(save_feature_dir)
            #     for n, f in enumerate(features[:, 0]):
            #         if np.isnan(f):
            #             features = features[:n, :]
            #             np.nan_to_num(features)
            #             break
            #             # features[np.isnan(features)] = 0
            #     features[:, 1] = list(seq)
            #     # features[:, 5] = features[:, -1]/100
            #     features[:, 2] = (features[:, 2] - min(features[:, 2])) / (max(features[:, 2]) - min(features[:, 2]))
            #     features[:, -1] = scale(features[:, -1].astype(np.float64))
            #     DataFrame(features[:, 1:]).to_csv(os.path.join(save_feature_dir, name + '_features'), sep=' ', header=0,
            #                                       index=0)
            # elif generate_features_table_csv:
            #     DataFrame(attn_list).to_csv(save_dir + '/' + name.lower() + '_features.csv', sep=',', header=1, index=0)
            #     continue

            cords = np.array(coordinate, dtype=np.float32)
            dist0 = compute_pairwise_distances(cords, cords)
            # dist = 1 / (1 + dist / d0)
            # if only_pocket_amino:
            #     seq, dist, features = To_pocket_amino(dist, features[:, 1:])
            #     DataFrame(features).to_csv(os.path.join(save_feature_dir, name + '_features'), sep=' ', header=0,
            #                                index=0)

            dist, avg_edge_degree, avg_edge = constrcut_hypergraph(dist0, mu=5.8, k=10, w=5, tao=8)

            x, y = np.nonzero(dist)
            value = dist[x, y]
            contactmap_coo = np.vstack((x, y, value)).T.astype(np.float32)
            dist = add_name_seq(dist_map=contactmap_coo, protein_name=name, seq=seq)
            # save_contactMap(os.path.join(save_dir, name), dist)
            avg_edges_degree.append(avg_edge_degree)
            avg_nodes.append(len(seq))
            avg_edges.append(avg_edge)

            # contactmap_coo_dict[name] = contactmap_coo

    print("protein contactmap  processing successfully")
    return np.array(avg_nodes), np.array(avg_edges_degree), np.array(avg_edges)


def constrcut_hypergraph(dist, mu=5.8, k=10, w=5, tao=8):
    dist = np.int8(dist < mu)
    edge_degree = 0
    incidence = []
    backbone = []
    nonzero_index_list = []
    for e in dist:
        e_n = sum(e)
        if e_n < 2 or e_n > k:
            continue
        nonzero_index = np.nonzero(e)[0]
        if max(nonzero_index) - min(nonzero_index) + 1 < tao:
            continue
        should_simplify = simplify(nonzero_index_list, nonzero_index)
        if should_simplify:
            continue
        nonzero_index_list.append(nonzero_index)
        incidence.append(e)
        edge_degree += e_n
    n1 = len(incidence)
    for i in range(0, len(dist) - w + 1):
        e_backbone = np.zeros(len(dist))
        e_backbone[i:i + w] = 1
        incidence.append(e_backbone)
    avg_edge_degree = ((len(incidence) - n1) * w + edge_degree) / len(incidence)
    # incidence = incidence.extend(backbone)
    incidence = np.vstack(incidence)
    avg_edge = len(incidence)
    return incidence.T, avg_edge_degree, avg_edge


def simplify(nonzero_index_list, nonzero_index):
    for e_index in nonzero_index_list:
        if set(e_index).issubset(set(nonzero_index)) or set(nonzero_index).issubset(set(e_index)):
            return True
    return False


if __name__ == '__main__':
    dit = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
           'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
           'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
           'HIP': 'H', 'HIE': 'H', 'TPO': 'T', 'HID': 'H', 'LEV': 'L', 'MEU': 'M', 'PTR': 'Y',
           'GLV': 'E', 'CYT': 'C', 'SEP': 'S', 'HIZ': 'H', 'CYM': 'C', 'GLM': 'E', 'ASQ': 'D',
           'TYS': 'Y', 'CYX': 'C', 'GLZ': 'G'}

    mass_dict = []  # load_mass(path='./mass.txt')

    path = r'..\data\PROTEINS\gpcr'
    contactmap_save_dir = 'dataset/protein/gpcr'
    avg_nodes, avg_edges_degree,avg_edges = generate_contactmap(mass_dict=mass_dict, dit=dit, protein_3Dpath=path,
                                                      save_dir=contactmap_save_dir, d0=3.8,
                                                      generate_features_table_csv=False)
    print(f'avg_nodes={avg_nodes.mean()}+-{avg_nodes.std()}\n'
          f'avg_edges={avg_edges.mean()}+-{avg_edges.std()}\n'
          f'avg_edges_degree={avg_edges_degree.mean()}+-{avg_edges_degree.std()}')
    # seq1 = generate_new_contactmap(mass_dict=mass_dict, dit=dit, protein_3Dpath=path,
    #                                contactmap_coo_dict=contactmap_coo_dict,
    #                                save_dir=contactmap_save_dir, d0=3.8, generate_features_table_csv=False)

    # two_seq_compare(seq1, seq2)
