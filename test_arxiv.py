from sklearn.model_selection import StratifiedKFold
from dataset.arxiv_dataset import ArXiv
import numpy as np
import random

if __name__ == '__main__':
    seed = 1234
    num_fold = 2
    arxiv_dataset = ArXiv(verbose=True, force_rebuild=True)
    graphs_list, num_classes, fea_dim = [
        arxiv_dataset.get_graphs(),
        arxiv_dataset.num_classes,
        arxiv_dataset.feature_dim]
    labels = graphs_list['labels']
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    train_split, test_split = list(skf.split(np.zeros(len(labels)), labels))[num_fold]
    train_graph_list, test_graph_list = graphs_list['callback'](train_split, test_split)
    print(f'#train = {len(train_graph_list)},\n#test = {len(test_graph_list)}\n')
    for _ in range(10):
        data = train_graph_list[random.randint(0, len(train_graph_list))]
        g = data.g
        g_row_set = set(g.row)
        g_col_set = set(g.col)
        print(f'#node = {len(g_row_set)}, #edge: {len(g_col_set)}')
        assert 0 == min(g_row_set) and max(g_row_set) == len(g_row_set) - 1
        assert 0 == min(g_col_set) and max(g_col_set) == len(g_col_set) - 1
        assert len(data.node_features) == len(g_row_set)
