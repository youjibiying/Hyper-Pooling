import json
from pathlib import Path
from time import time
from itertools import chain
import random
import socket
import socks
import numpy as np
import os
from queue import Queue
from .s2vgraph import S2VGraph
from scipy.sparse import coo_matrix

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# Set CPU as available physical device
# physical_devices = tf.config.list_physical_devices('GPU')
# print(f'physical_devices: {physical_devices}')
# tf.config.set_visible_devices(physical_devices[0:2], 'GPU')
# Force use CPU
# https://github.com/tensorflow/tensorflow/issues/31135
import tensorflow as tf  # noqa
import tensorflow_hub as hub  # noqa


class ArxivPreprocess:
    def __init__(self, verbose=False, min_edge=20, max_edge=400, min_width=3,
                 min_graph=500, proxy=True, raw_data_path='dataset/arXiv/decompressed',
                 target_path='dataset/arXiv', filename='arxiv_hypergraph_unsplited.npz',
                 sentence_model_path='https://tfhub.dev/google/universal-sentence-encoder/4'):
        # params
        self.verbose = verbose
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.min_graph = min_graph
        self.raw_data_path = raw_data_path
        self.min_width = min_width
        self.target_path = target_path
        self.sentence_model_path = sentence_model_path
        self.filename = filename
        if proxy:
            self._set_proxy()
        self._metadata = []
        self._metadata_id_dict = None
        self._citation = None
        self._category_target = None
        self.dataset_target_final = None
        self._dataset = None
        self._dataset_stat = None
        self.graphs = []
        self.corpus = None
        self.dataset_downsampling = None

    @staticmethod
    def _set_proxy():
        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
        socket.socket = socks.socksocket

    @staticmethod
    def _get_paper_text(meta):
        return str(meta['title'].replace('\n', ' ') + '. ' + meta['abstract'].replace('\n', ' '))

    def _check_valid_id(self, pid):
        return pid in self._metadata_id_dict and pid in self._citation

    def _save_npz(self):
        # 导出文件：graphs: [..., [pid_i, edges_i], ...], 其中 pid_i 为第 i 个 graph 构造的时候的起始论文的 ArXiv ID，
        # edges_i 为  [..., [pid_j0, ..., pid_jn], ...] 为第 i 个 graph 的超边集，每一条边长度不一样，例如第 j 条超边
        # 长度为 jn，pid_j0 表示论文的 ArXiv ID。
        # dataset: {class: [gid_0, ...], ...}，共有五个类：{'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.DS'},
        # 每个类对应一个数组，数组里面就是相应的 graph 的索引。
        # corpus: {pid: paper_embd, ...}, 为 ArXiv ID 对应 paper 的标题和摘要的 embedding（512-D vector）。
        st = time()
        target = Path(self.target_path, self.filename)
        np.savez_compressed(target,
                            graphs=self.graphs,
                            dataset=self.dataset_downsampling,
                            corpus=self.corpus)
        print(f'File written into {target.resolve()}' +
              f'\n(size={os.path.getsize(target)})' +
              f' with {time() - st}s.')

    def get_meta(self, pid, field=None):
        if field:
            return self._metadata[self._metadata_id_dict[pid]][field]
        else:
            return self._metadata[self._metadata_id_dict[pid]]

    def _load_data(self):
        st = time()
        print('Start loading raw data...')
        with open(list(Path(self.raw_data_path).glob('internal-references*'))[0], 'r') as f:
            self._citation = json.loads(f.readlines()[0])
            if self.verbose:
                print(f'Done with {time() - st}s\n',
                      'Citation:\n', self._citation['1307.7980'])
        st = time()
        with open(list(Path(self.raw_data_path).glob('arxiv-metadata-oai*'))[0], 'r') as f:
            for line in f.readlines():
                self._metadata.append(json.loads(line))
            self._metadata_id_dict = {v['id']: i for i, v in enumerate(self._metadata)}
            if self.verbose:
                print(f'Done with {time() - st}s.')
                print('Metadata:\n', '\n\n'.join(map(lambda x: str(x), self._metadata[:1])))
                print(self.get_meta(self._citation['1307.7980'][2]))
        print(f'Done, metadata: {len(self._metadata)}, citation: {len(self._citation)}')

    def _category_stat(self):
        category_stat = {}
        category_stat_coarse = {}
        if self.verbose:
            print('Start category statistic...')
        for p in self._citation:
            if p in self._metadata_id_dict:
                c = self.get_meta(p, 'categories')
                if ' ' not in c:
                    # if single category
                    if c not in category_stat:
                        category_stat[c] = 1
                    else:
                        category_stat[c] += 1
                    c = c.split('.')[0]
                    if c not in category_stat_coarse:
                        category_stat_coarse[c] = 1
                    else:
                        category_stat_coarse[c] += 1
        if self.verbose:
            print(f'category_stat (len={len(category_stat)}): {category_stat}' +
                  f'({len(list(filter(lambda x: x[1] >= 2000, category_stat.items())))}' +
                  f' keys with values >= 2000)\n\ncategory_stat_coarse (len=' +
                  f'{len(category_stat_coarse)}): {category_stat_coarse}')
        category_target_stat = list(filter(lambda x: x[0].startswith("cs.") and x[1] >= 2000,
                                           category_stat.items()))
        self._category_target = set(map(lambda x: x[0], category_target_stat))
        print('Done, category_stat of subclasses of "cs" with more than 2000 papers:\n' +
              f'{category_target_stat}\n')

    def _construct_graph(self):
        dbg = {c: {'nodes': [], 'uni_nodes': [], 'edge': []} for c in self._category_target}
        self._dataset = {c: [] for c in self._category_target}
        cnt = 0
        cnt_empty_ref = 0
        cnt_fail = 0
        cnt_self_ref = 0
        st = time()
        print('start construct graphs.')
        for pid, refs in filter(lambda x: len(x[1]) >= self.min_width, self._citation.items()):
            # remove self ref
            if pid in refs:
                cnt_self_ref += 1
                refs = [r for r in refs if r != pid]
            cnt += 1
            if cnt % 40000 == 0:
                print(f'cnt: {cnt / 10000:.2f}w/135w, with {(time() - st) / cnt * 10000:.1f}s/10000')
            category = self.get_meta(pid, 'categories')
            if category not in self._category_target:
                continue
            refs = list(filter(self._check_valid_id, refs))
            if len(refs) == 0:
                cnt_empty_ref += 1
                continue
            q = Queue()
            for r in refs:
                q.put(r)
            # 已经在边里面的论文不用继续遍历其引用，否则会引起回环！
            exists_node_set = set(refs)
            # one paper with its citations construct one edge
            edges = [[pid, ] + refs, ]
            while q.qsize() > 0 and len(edges) <= self.max_edge:
                curr = q.get()
                refs = set(filter(self._check_valid_id, self._citation[curr]))
                edges.append([curr, ] + list(refs))
                refs = set([r for r in refs if r not in exists_node_set])
                exists_node_set |= refs
                for r in refs:
                    q.put(r)
            if len(edges) >= self.min_edge:
                self.graphs.append([pid, edges])
                self._dataset[category].append(len(self.graphs) - 1)
                nodes = list(chain.from_iterable(edges))
                dbg[category]['nodes'].append(len(nodes))
                dbg[category]['uni_nodes'].append(len(set(nodes)))
                dbg[category]['edge'].append(len(edges))
            else:
                cnt_fail += 1
        self._dataset_stat = {k: len(v) for k, v in self._dataset.items()}
        self.dataset_target_final = set([k for k, v in self._dataset_stat.items() if v >= self.min_graph])
        dbg = {c: dict(map(lambda x: (x[0], round(sum(x[1]) / len(x[1]), 2)), dbg[c].items())) for c in dbg}
        if self.verbose:
            print(f'cnt_fail = {cnt_fail}, cnt_empty_ref = {cnt_empty_ref}, cnt_self_ref: {cnt_self_ref}')
        print(f'#graph = {len(self.graphs)}\n\ndataset: {self._dataset_stat}\n\n' +
              f'{dbg}\n')

    def _generate_node_features(self):
        embed = hub.load(self.sentence_model_path)
        print(f'dataset_target_final: {self.dataset_target_final}' +
              '\nStart generate node features...')
        # if #samples >= 1000, then subsampling to 700
        self.dataset_downsampling = {c: (random.sample(d, len(d)) if self._dataset_stat[c] <= 1000 else
                                         random.sample(d, 700)) for c, d in self._dataset.items()
                                     if c in self.dataset_target_final}
        print(f'dataset: {[[c, len(v)] for c, v in self.dataset_downsampling.items()]}')

        st = time()
        dataset_pids = set([self.graphs[i][0] for i in list(chain.from_iterable(
            self.dataset_downsampling.values()))])
        # Get node features
        pids = list(set(chain.from_iterable([set(chain.from_iterable(x[1])) for x in self.graphs
                                             if x[0] in dataset_pids])))
        # decrease memory usage
        self.graphs = [g if g[0] in dataset_pids else [g[0], None] for g in self.graphs]
        self.corpus = {p: self._get_paper_text(self.get_meta(p)) for p in pids}
        print(f'#corpus={len(self.corpus)}')
        batch_size = 2048
        for idx in range(int(np.ceil(len(self.corpus) / batch_size))):
            curr_range = pids[idx * batch_size: (idx + 1) * batch_size]
            # => 512-d feature vector
            emb = embed([self.corpus[i] for i in curr_range]).numpy()
            for i, pid in enumerate(curr_range):
                self.corpus[pid] = emb[i]
        print(f'Done embedding {len(self.corpus)} paper corpus with {time() - st}s.')

    def run(self):
        self._load_data()
        self._category_stat()
        self._construct_graph()
        self._generate_node_features()
        self._save_npz()


class ArXiv:
    """
    用于加载已经预处理好的数据集，提供一个简单的训练集/测试集分割接口。
    数据集为 S2VGraph 的列表。
    Source:
    https://www.kaggle.com/Cornell-University/arxiv
    Ref:
    [1]: http://arxiv.org/abs/1905.00075v1
    """
    def __init__(self, data_path='dataset/arXiv/',
                 filename='arxiv_hypergraph_unsplited.npz',
                 verbose=False, force_rebuild=False):
        path = Path(data_path, filename)
        if not path.exists() or force_rebuild:
            print(f'Dataset file {path.resolve()} does not exists, ' +
                  'now generate it.')
            ArxivPreprocess(verbose=verbose).run()
        print('Loading dataset...')
        data = np.load(path, allow_pickle=True)
        self.graphs = data['graphs']
        self.dataset = data['dataset'].item()
        self.num_classes = len(self.dataset)
        c2id = {k: i for i, k in enumerate(self.dataset.keys())}
        self.dataset = list(chain.from_iterable(
                [[(c2id[c], gid) for gid in v] for c, v in self.dataset.items()]))
        self.corpus = data['corpus'].item()
        self.feature_dim = self.corpus[list(self.corpus.keys())[0]].shape[0]
        print(f'Done, class to label: {c2id}')
        self.verbose = verbose

    def get_labels(self):
        return list(zip(*self.dataset))[0]

    def _construct_hypergraph(self, edges, label, test_pids, test=False):
        # construct hypergraph H with shape (N, E)
        pid2nid = {pid: i for i, pid in enumerate(
            set(chain.from_iterable(edges)) - test_pids)}
        matrix_h = list(chain.from_iterable([
            [[pid2nid[p], e_id] for p in edge if (test or (p not in test_pids))] for e_id, edge in enumerate(edges)]))
        matrix_h = sorted(matrix_h, key=lambda x: x[0])
        row = [i[0] for i in matrix_h]
        col = [i[1] for i in matrix_h]
        # currently all elements in H are 1
        data = [1.] * len(row)
        node_features = np.stack([self.corpus[i[0]] for i in sorted(
            [[k, v] for k, v in pid2nid.items()], key=lambda x: x[1])])
        coo = coo_matrix((data, (row, col)), shape=(max(row) + 1, max(col) + 1))
        return S2VGraph(coo, label, node_features)

    def split(self, train_idx, test_idx):
        st = time()
        dataset_train = [v for i, v in enumerate(self.dataset) if i in train_idx]
        dataset_test = [v for i, v in enumerate(self.dataset) if i in test_idx]
        # In order to prevent a leaking of the test set into the training set,
        # using the train/test partition that we omitted citations from articles
        # in the training set which connect to the test set, but retained citations
        # in the test set which connect to the training set.
        test_pids = set([self.graphs[pid][0] for _, pid in dataset_test])
        train_graphs = list([self._construct_hypergraph(
            self.graphs[d][1], c, test_pids) for c, d in dataset_train])
        test_graphs = list([self._construct_hypergraph(
            self.graphs[d][1], c, test_pids) for c, d in dataset_test])
        print(f'Construct hypergraph done with {time() - st}s.')
        return train_graphs, test_graphs

    def get_graphs(self):
        # coo-matrix (sparse matrix) of H with shape (N, E): row, col, data with shape (k, 1)
        # node features: 2D list, shape: (N, D)
        return {'labels': self.get_labels(),
                'callback': lambda train_idx, test_idx: self.split(train_idx, test_idx)}
