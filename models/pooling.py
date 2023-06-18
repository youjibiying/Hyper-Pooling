import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.layers import HGraphConvolutionBS
import torch
import torch.nn as nn
import numpy as np

'''
class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=HGraphConvolutionBS,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1) #X_idx \odot Z_idx
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0)) # 过滤掉 分数低的节点对应的edge

        return x, edge_index, edge_attr, batch, perm
'''


def norm_g(adj):  # todo norm -> hypergraph norimization
    degrees = torch.sum(adj, 1)
    adj = adj / degrees
    return adj


def nrom_hypergraph( H, variable_weight=False): # todo norm -> hypergraph norimization
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    # H = np.array(H)
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = torch.ones(n_edge).type_as(H)  # 使用权重为1
    # the degree of the node
    DV = torch.sum(H * W,
                   dim=1)  # [2012]数组*是对应位置相乘，矩阵则是矩阵乘法 https://blog.csdn.net/TeFuirnever/article/details/88915383
    # the degree of the hyperedge
    DE = torch.sum(H, dim=0)  # [4024]

    # invDE = torch.mat(torch.diag(torch.pow(DE, -1)))
    invDE = torch.diag(torch.pow(DE, -1))
    invDE[torch.isinf(invDE)] = 0  # D_e ^-1
    invDV = torch.pow(DV, -0.5)
    invDV[torch.isinf(invDV)] = 0
    DV2 = torch.diag(invDV)  # D_v^-1/2

    W = torch.diag(W)
    # H = np.mat(H)
    HT = H.t()

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2.mm(H).mm(W).mm(invDE).mm(HT).mm(DV2)
        # G = DV2 * H * W * invDE * HT * DV2  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2
        return G

# def top_k_graph(scores, adj, h, k): # simple graph
#     num_nodes = adj.shape[0]
#     values, idx = torch.topk(scores, max(2, int(k * num_nodes)))  #
#     new_h = h[idx, :]
#     values = torch.unsqueeze(values, -1)
#     new_h = torch.mul(new_h, values)  # 对特征加权
#     un_g = adj.bool().float()
#     un_g = torch.matmul(un_g, un_g).bool().float()  # AA
#     un_g = un_g[idx, :]
#     un_g = un_g[:, idx]
#     adj = norm_g(un_g)
#     return adj, new_h, idx
def top_k_graph(scores, H, h, k):
    '''
     incidence(H) of hypergraph top-k
    '''
    num_nodes = H.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))  #
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)  # 对特征加权
    un_g = H.bool().float()
    # un_g = torch.matmul(un_g, un_g).bool().float()  # AA
    H = un_g[idx, :]
    # un_g = un_g[:, idx]
    G = nrom_hypergraph(H)
    return H, G, new_h, idx

class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)  # todo 这里的投影可以换成 HGCN, 学习拓扑结构作为分数(参考 SAGpooling)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, adj, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, adj, h, self.k)  # return  adj, new_h, idx


class HGraphUnet(nn.Module):

    def __init__(self, ks, in_features, hiddendim, activation, dropout, args, withbn=False, res=False):
        super(HGraphUnet, self).__init__()
        self.ks = ks
        self.args = args
        self.hiddendim = hiddendim
        self.bottom_hgcn = HGraphConvolutionBS(hiddendim, hiddendim, activation=activation, withbn=withbn,
                                               res=res, args=self.args)  #
        self.down_hgcns = nn.ModuleList()
        # self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        # self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_hgcns.append(HGraphConvolutionBS(hiddendim, hiddendim, activation, dropout))
            # self.up_gcns.append(HGraphConvolutionBS(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], hiddendim, dropout))
            # self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, h, H, G=None):  # G 为处理好的laplasion todo: 直接换成H
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        for i in range(self.l_n):
            h = self.down_hgcns[i](input=h, G=G)  # X=AXW
            adj_ms.append(G)
            down_outs.append(h)
            H, G, h, idx = self.pools[i](H, h)
            indices_list.append(idx)
        h = self.bottom_hgcn(input=h, G=G)
        # for i in range(self.l_n):
        #     up_idx = self.l_n - i - 1
        #     adj, idx = adj_ms[up_idx], indices_list[up_idx]
        #     adj, h = self.unpools[i](adj, h, down_outs[up_idx], idx)
        #     h = self.up_gcns[i](adj, h)
        #     h = h.add(down_outs[up_idx])
        #     hs.append(h)
        # h = h.add(org_h)
        # hs.append(h)
        return h  # hs

    def get_outdim(self):
        return self.hiddendim
