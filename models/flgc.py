import torch
import torch.nn as nn
import math

import torch.nn.functional as F


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(context, probs, nei_k=1, dim=1):
        """
        -dim =1 取行的topk 的值为1 ，dim=0 列topk为1
        """
        # binarized = (probs == torch.max(probs, dim=1, keepdim=True)[0]).float()
        binarized = torch.zeros(probs.shape).type_as(probs)
        if type(nei_k) == int:
            idx = probs.topk(k=nei_k, dim=dim)[1]
        else:
            idx = [probs[i].topk(k=int(k), dim=0)[1] for i, k in enumerate(nei_k)]
        # max_dist_edge = torch.topk(-probs, k=2, dim=dim)[1]
        # idx = max_dist_edge
        # idx=[torch.cat((max_dist_edge[i],idx[i]),dim=0) for i in range(len(probs))]
        for i, l in enumerate(idx):
            binarized[i, l] = 1

        context.save_for_backward(binarized)
        return binarized

    @staticmethod
    def backward(context, gradient_output):
        binarized, = context.saved_tensors
        gradient_output[binarized == 0] = 0
        return gradient_output, None, None, None  # backward() 梯度公式，它会返回和forward()输入一样多的值，每个值是对于输入的梯度。如果输入不需要梯度，可以返回None。


class Generate_G_from_H(nn.Module):

    def forward(self, H, variable_weight=False):
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

        # n_edge = H.shape[1]  # 4024
        # # the weight of the hyperedge
        # W = torch.ones((n_edge, 1), device=H.device)  # 使用权重为1
        # # the degree of the node
        # DV = torch.sum(torch.spmm(H, W),
        #                dim=1)  # mm矩阵乘法
        # # the degree of the hyperedge
        # DE = torch.sparse.sum(H, dim=0).to_dense()  # [4024]
        #
        # # invDE = torch.mat(torch.diag(torch.pow(DE, -1)))
        # invDE = torch.diag(torch.pow(DE, -1))
        # invDE[torch.isinf(invDE)] = 0  # D_e ^-1
        # invDV = torch.pow(DV, -0.5)
        # invDV[torch.isinf(invDV)] = 0
        # DV2 = torch.diag(invDV)  # D_v^-1/2
        #
        # W = torch.diag(W.squeeze(1))
        # # H = np.mat(H)
        # HT = H.t()
        #
        # if variable_weight:
        #     DV2_H = DV2 * H
        #     invDE_HT_DV2 = invDE * HT * DV2
        #     return DV2_H, W, invDE_HT_DV2
        # else:
        #     # G=torch.spmm(DV2,H.to_dense())
        #     # G= torch.spmm(G,W)
        #     # G= torch.spmm(G,invDE)
        #     # G= torch.spmm(G,HT.to_dense())
        #     # G= torch.spmm(G,DV2)
        #
        #     G = DV2.mm(H.to_dense()).mm(W).mm(invDE).mm(HT.to_dense()).mm(DV2)
        #     # G = DV2 * H * W * invDE * HT * DV2  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2
        #     return G.to_sparse()


class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """

    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=0)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()  # (N, k, d)
        conved = self.convKK(region_feats)  # (N, k*k, 1)
        multiplier = conved.view(N, k, k)  # (N, k, k)
        multiplier = self.activation(multiplier).t()  # softmax along last dimension
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats


class cosine(nn.Module):
    def __init__(self):
        super(cosine, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, h1, h2=None):
        # M = h1.size()[0]
        N = h1.size()[0]
        similarity = []
        # similarity = self.cos(h1.repeat(1, N).view(M * N, -1), h2.repeat(M, 1)).view(M, N)
        for h in h1:
            s = self.cos(h.expand_as(h1), h1).view(1, N)
            similarity.append(s)
        similarity = torch.cat(similarity, dim=0)
        return similarity


class BilinearAttention(nn.Module):

    def __init__(self, k_dim, q_dim):
        super(BilinearAttention, self).__init__()

        # attention
        self.W = nn.Linear(q_dim, k_dim, bias=False)  # bidirectional

    def forward(self, query, values):
        query = self.W(query)
        scores = torch.matmul(query, values.transpose(0, 1))  # [B,1,H]*[B,H,T] -> [B,1,T]
        return scores


class Self_attn(nn.Module):

    def __init__(self, k_dim, q_dim):
        super(Self_attn, self).__init__()

        # attention

        self.W_q = nn.Linear(q_dim, k_dim, bias=False)  # bidirectional
        self.W_k = nn.Linear(q_dim, k_dim, bias=False)  # bidirectional

    def forward(self, query, values):
        query = self.W_q(query)
        values = self.W_k(values)
        scores = torch.matmul(query, values.transpose(0, 1))  # [B,1,H]*[B,H,T] -> [B,1,T]
        return scores


class ScaledDotProductAttention(nn.Module):
    # def __init__(self):
    #     super(ScaledDotProductAttention, self).__init__()
    #     # attention
    #     self.softmax = nn.Softmax(dim=1)
    def forward(self, query):
        scores = torch.matmul(query, query.transpose(0, 1)) / math.sqrt(query.shape[1])  # Q*V.T/sqrt(d_k)
        return scores


class AdjConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """

    def __init__(self, in_feature=None, hidden=None, e_n=None, only_G=True, theta=0.01):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        # self.trans = Transform(dim_in=e_n, k=1)
        # (N, k, d) -> (N, k, d)
        # self.convK1 = nn.Conv1d(1, e_n, 1)  # (N, k, d) -> (N, 1, d)
        # self.MLP = nn.Sequential(nn.Linear(e_n, hidden),
        #                          nn.LeakyReLU(negative_slope=0.01),
        #                          nn.Dropout(p=0.5),
        #                          nn.Linear(hidden, e_n),
        #                          nn.Sigmoid())
        self.linear = nn.Linear(in_feature, hidden, bias=False)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(hidden, hidden))
        self.W_v = nn.Linear(in_feature, hidden, bias=False)  # bidirectional
        self.w_o = nn.Linear(hidden, 1)
        self._generate_G_from_H = Generate_G_from_H()
        self.use_binarize = False
        self.only_G = only_G
        self.theta = theta
        self.cosine = cosine()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.Scales = ScaledDotProductAttention()
        self.self_attn = Self_attn(hidden, in_feature)  # Self_attn(in_feature, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.binarize = Binarize.apply

        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def Euclidean_dist(self, x):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """

        m, n = x.size(0), x.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(x, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
        dist.addmm_(x, x.t(), beta=1, alpha=-2)
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def hadamard_power(self, x, y):
        scores = []
        N = y.shape[0]
        M = x.size(0)
        x = x.repeat(1, N).view(M * N, -1)  #
        y = y.repeat(M, 1)
        dist = (x - y) ** 2

        scores = self.w_o(dist).view(M, N)
        # for i in x:
        #     l = torch.dist(i.expand_as(y), y)
        #     scores.append(l)
        # scores = torch.stack(scores)
        # socres=(i.expand(y.shape[0],1)-y)**2
        # self.W_o(socres)
        return scores

    def forward(self, adj=None, G=None, feats=None, kn=10, num=None):
        """
        :param adj: (v_n, e_n)
        :return: (v_n, e_n)
        """
        adj = adj.t()  # (e_n, v_n)

        ## 1. self attention
        # x = torch.tanh(torch.matmul(self.w1, feats.transpose(1, 0)))
        # x = torch.matmul(self.w2, x)
        # # attn = torch.nn.functional.softmax(x, dim=-1)
        # # feats = torch.matmul(attn, feats)
        # transformed_adj = -torch.matmul(x.transpose(0, 1),x)
        # transformed_adj = torch.exp(-transformed_adj / (2 * transformed_adj.std() ** 2))

        # x = self.linear(feats)
        # transformed_adj = -torch.matmul(x.transpose(0, 1),x)

        # transformed_adj = self.cosine(feats, feats)
        # transformed_adj = self.sigmoid(self.Scales(feats)) # 0.8136

        ## 2.agcn
        # feats = self.linear(feats)  # todo: 在同一个边中加入离当前边中心点最远的点
        # transformed_adj = self.Euclidean_dist(feats)
        # if transformed_adj.std()==0:
        #     raise ValueError
        # transformed_adj = torch.exp(-transformed_adj / (2 * transformed_adj.std() ** 2))
        # transformed_adj = torch.exp(-transformed_adj / (2 * 20 ** 2))
        ##

        # transformed_adj = self.softmax(self.Scales(feats))  # 0.8147
        # transformed_adj = self.MLP(transformed_adj)
        # transformed_adj = self.MLP(adj)

        # 求边的中心
        e_center = feats.t().mm(adj.t()) / adj.sum(1)
        # attn
        feats = self.W_v(feats)

        # scores = self.self_attn(feats, feats)
        scores = self.Scales(feats)
        attn = self.softmax(scores)

        s = self.linear(e_center.t())
        d = torch.matmul(attn, feats)

        s = self.layer_norm1(s)
        d = self.layer_norm2(d)
        transformed_adj = self.hadamard_power(s, d)
        # if torch.any(torch.isnan(transformed_adj)):
        #     print(transformed_adj)
        # transformed_adj= self.sigmoid(transformed_adj) # citaton cora 中挺有效果
        transformed_adj = torch.exp(-transformed_adj / (2 * 20 ** 2))

        # binarized = self.binarize(transformed_adj, adj.sum(1), 1)
        # add_edge_adj = torch.max(adj, binarized)  # 加边
        # transformed_adj = transformed_adj * add_edge_adj  # 加权

        # core_edge_adj = binarized * adj
        # transformed_adj = transformed_adj * (self.theta * add_edge_adj + (1 - self.theta) * core_edge_adj)

        # binarized = binarized.to_sparse()
        # transformed_adj =  add_edge_adj

        # transformed_adj = self.theta * (binarized + adj) + (1 - 2 * self.theta) * binarized * adj
        # transformed_adj = transformed_adj * binarized  # 加权
        # transformed_adj = self.MLP(similarity) * adj # 加权
        # transformed_adj = transformed_adj.squeeze(1)
        # attention = torch.where(adj > 0, e, zero_vec) #大于0的位置取e,小于0的取-9e15 :todo: 增加边

        # transformed_adj = self.convK1(transformed_adj)  # (N, 1, d)
        if self.use_binarize:
            attn = torch.nn.functional.softmax(transformed_adj, dim=-1)
            binarized = self.binarize(attn, kn, 1)
            H = binarized.t()  # (v_n, e_n)
            self.adj = H
            # H = (binarized * self.vertex_assignment_map).t()
            G_new = self._generate_G_from_H(H)
        else:
            G_new = self._generate_G_from_H(transformed_adj.t())
            # G = F.sigmoid(transformed_adj)
            self.adj = G_new  # 保存看看学的结果
        if num is not None:
            theta = 1 - (1 - self.theta) * (math.cos(math.pi * (num - 1) / 10) + 1) / 2
        else:
            theta = self.theta
        # theta = math.log((1 - self.theta) / num + 1) # 0.9
        G_new = (1 - theta) * G + theta * G_new
        # self.adj = self.vertex_assignment_mp.detach().data# 保存看看学的结果

        if self.only_G:
            # x = torch.matmul(attn, X)
            return G_new.float()  # x, attn
        return torch.spmm(G_new, feats)  # nearest_feature


class AdjConv1(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """

    def __init__(self, in_feature=None, hidden=None, e_n=None, only_G=True, theta=0.01):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        # self.trans = Transform(dim_in=e_n, k=1)
        # (N, k, d) -> (N, k, d)
        # self.convK1 = nn.Conv1d(1, e_n, 1)  # (N, k, d) -> (N, 1, d)
        # self.MLP = nn.Sequential(nn.Linear(e_n, hidden),
        #                          nn.LeakyReLU(negative_slope=0.01),
        #                          nn.Dropout(p=0.5),
        #                          nn.Linear(hidden, e_n),
        #                          nn.Sigmoid())
        self.linear = nn.Linear(in_feature, hidden, bias=False)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(hidden, hidden))
        self.W_v = nn.Linear(in_feature, hidden, bias=False)  # bidirectional
        self.w_o = nn.Linear(hidden, 1)
        self._generate_G_from_H = Generate_G_from_H()
        self.use_binarize = False
        self.only_G = only_G
        self.theta = theta
        self.cosine = cosine()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.Scales = ScaledDotProductAttention()
        self.self_attn = Self_attn(hidden, in_feature)  # Self_attn(in_feature, hidden)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.binarize = Binarize.apply

        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def Euclidean_dist(self, x):
        """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
        """

        m, n = x.size(0), x.size(0)
        # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        # yy会在最后进行转置的操作
        yy = torch.pow(x, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
        dist.addmm_(x, x.t(), beta=1, alpha=-2)
        # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def hadamard_power(self, x, y):
        scores = []
        N = y.shape[0]
        M = x.size(0)
        x = x.repeat(1, N).view(M * N, -1)  #
        y = y.repeat(M, 1)
        dist = (x - y) ** 2

        scores = self.w_o(dist).view(M, N)
        # for i in x:
        #     l = torch.dist(i.expand_as(y), y)
        #     scores.append(l)
        # scores = torch.stack(scores)
        # socres=(i.expand(y.shape[0],1)-y)**2
        # self.W_o(socres)
        return scores

    def forward(self, adj=None, G=None, feats=None, kn=10, num=None):
        """
        :param adj: (v_n, e_n)
        :return: (v_n, e_n)
        """
        adj = adj.t()  # (e_n, v_n)

        ## 1. self attention
        # x = torch.tanh(torch.matmul(self.w1, feats.transpose(1, 0)))
        # x = torch.matmul(self.w2, x)
        # # attn = torch.nn.functional.softmax(x, dim=-1)
        # # feats = torch.matmul(attn, feats)
        # transformed_adj = -torch.matmul(x.transpose(0, 1),x)
        # transformed_adj = torch.exp(-transformed_adj / (2 * transformed_adj.std() ** 2))

        # x = self.linear(feats)
        # transformed_adj = -torch.matmul(x.transpose(0, 1),x)

        # transformed_adj = self.cosine(feats, feats)
        # transformed_adj = self.sigmoid(self.Scales(feats)) # 0.8136

        ## 2.agcn
        # feats = self.linear(feats)  # todo: 在同一个边中加入离当前边中心点最远的点
        # transformed_adj = self.Euclidean_dist(feats)
        # if transformed_adj.std()==0:
        #     raise ValueError
        # transformed_adj = torch.exp(-transformed_adj / (2 * transformed_adj.std() ** 2))
        # transformed_adj = torch.exp(-transformed_adj / (2 * 20 ** 2))
        ##

        # transformed_adj = self.softmax(self.Scales(feats))  # 0.8147
        # transformed_adj = self.MLP(transformed_adj)
        # transformed_adj = self.MLP(adj)

        # 求边的中心
        e_center = feats.t().mm(adj.t()) / adj.sum(1)
        # attn
        scores = self.self_attn(feats, feats)
        attn = self.softmax(scores)

        s = self.linear(e_center.t())
        feats = self.W_v(feats)
        d = torch.matmul(attn, feats)

        s = self.layer_norm1(s)
        d = self.layer_norm2(d)
        transformed_adj = self.hadamard_power(s, d)
        # if torch.any(torch.isnan(transformed_adj)):
        #     print(transformed_adj)
        # transformed_adj= self.sigmoid(transformed_adj) # citaton cora 中挺有效果
        transformed_adj = torch.exp(-transformed_adj / (2 * 20 ** 2))

        # binarized = self.binarize(transformed_adj, adj.sum(1), 1)
        # add_edge_adj = torch.max(adj, binarized)  # 加边
        # transformed_adj = transformed_adj * add_edge_adj  # 加权

        # core_edge_adj = binarized * adj
        # transformed_adj = transformed_adj * (self.theta * add_edge_adj + (1 - self.theta) * core_edge_adj)

        # binarized = binarized.to_sparse()
        # transformed_adj =  add_edge_adj

        # transformed_adj = self.theta * (binarized + adj) + (1 - 2 * self.theta) * binarized * adj
        # transformed_adj = transformed_adj * binarized  # 加权
        # transformed_adj = self.MLP(similarity) * adj # 加权
        # transformed_adj = transformed_adj.squeeze(1)
        # attention = torch.where(adj > 0, e, zero_vec) #大于0的位置取e,小于0的取-9e15 :todo: 增加边

        # transformed_adj = self.convK1(transformed_adj)  # (N, 1, d)
        if self.use_binarize:
            attn = torch.nn.functional.softmax(transformed_adj, dim=-1)
            binarized = self.binarize(attn, kn, 1)
            H = binarized.t()  # (v_n, e_n)
            self.adj = H
            # H = (binarized * self.vertex_assignment_map).t()
            G_new = self._generate_G_from_H(H)
        else:
            G_new = self._generate_G_from_H(transformed_adj.t())
            # G = F.sigmoid(transformed_adj)
            self.adj = G_new  # 保存看看学的结果
        if num is not None:
            theta = 1 - (1 - self.theta) * (math.cos(math.pi * (num - 1) / 10) + 1) / 2
        else:
            theta = self.theta
        # theta = math.log((1 - self.theta) / num + 1) # 0.9
        G_new = (1 - theta) * G + theta * G_new
        # self.adj = self.vertex_assignment_mp.detach().data# 保存看看学的结果

        if self.only_G:
            # x = torch.matmul(attn, X)
            return G_new.float()  # x, attn
        return torch.spmm(G_new, feats)  # nearest_feature


class Attention(nn.Module):
    """
       Self Attention Layer
       Given $X\in \mathbb{R}^{n \times in_feature}$, the attention is calculated by: $a=Softmax(W_2tanh(W_1X))$, where
       $W_1 \in \mathbb{R}^{hidden \times in_feature}$, $W_2 \in \mathbb{R}^{out_feature \times hidden}$.
       The final output is: $out=aX$, which is unrelated with input $n$.
    """

    def __init__(self, *, in_feature, hidden, out_feature, only_G=True):
        """
        The init function.
        :param hidden: the hidden dimension, can be viewed as the number of experts.
        :param in_feature: the input feature dimension.
        :param out_feature: the output feature dimension.
        """
        super(Attention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(out_feature, hidden))
        self.w = torch.nn.Parameter(torch.FloatTensor(in_feature, in_feature))  # binear atten
        self.only_G = only_G
        self._generate_G_from_H = Generate_G_from_H()
        self.binarize = Binarize.apply
        self.cosine = cosine()
        self.Bili = BilinearAttention(in_feature, in_feature)
        self.Scales = ScaledDotProductAttention()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Use xavier_normal method to initialize parameters.
        """
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)
        nn.init.eye_(self.w)

    def forward(self, adj=None, feats=None, kn=10, ):
        """
        The forward function.
        :param X: The input feature map. $X \in \mathbb{R}^{n \times in_feature}$.
        :return: The final embeddings and attention matrix.
        """
        ## 1. self attention
        # x = torch.tanh(torch.matmul(self.w1, feats.transpose(1, 0)))
        # x = torch.matmul(self.w2, x)

        # x = torch.matmul(self.w, feats.t())
        # x = -torch.matmul(feats, feats.t())
        # x = torch.matmul(feats, x)

        x = self.cosine(feats, feats)  # 可以起作用
        # x=self.Bili(feats,feats) # 没用
        # x=self.Scales(feats,feats) # meiyong
        attn = torch.nn.functional.softmax(x, dim=-1)
        binarized = self.binarize(attn, kn, 1)
        H = binarized.t()  # kn ==list
        # H = (binarized * self.vertex_assignment_map).t()
        G = self._generate_G_from_H(H)
        self.adj = G  # 保存看看学的结果
        # self.adj = self.vertex_assignment_map.detach().data# 保存看看学的结果

        # return torch.spmm(G, feats)  # nearest_feature

        # x = torch.matmul(attn, feats)
        if self.only_G:
            return G  # x, attn

        return torch.spmm(G, feats)  # nearest_feature


class Flgc2d(nn.Module):
    def __init__(self, v_n=None, e_n=None, init_dist=None, only_G=False, theta=0):
        super().__init__()
        self.only_G = only_G
        if init_dist is not None:
            self.vertex_assignment_map = nn.Parameter(init_dist)  # [6,10]
        else:
            self.vertex_assignment_map = nn.Parameter(torch.Tensor(e_n, v_n))  # [6,10]
            nn.init.normal_(self.vertex_assignment_map)
        self.theta = theta
        self.binarize = Binarize.apply
        # self.cosine = cosine()

        self._generate_G_from_H = Generate_G_from_H()

    def forward(self, adj=None, G=None, feats=None, kn=10, num=None):
        """

        :param kn: 每条超边有kn个vertexs
         :return: [v_n,e_n] 关联矩阵
        """
        probs = torch.softmax(self.vertex_assignment_map, dim=1)

        # x = self.cosine(feats, feats)  # 可以起作用
        probs = torch.nn.functional.softmax(probs, dim=-1)

        binarized = self.binarize(probs, kn, 1)
        H = binarized.t()  # kn ==list
        # H = (binarized * self.vertex_assignment_map).t()
        G_new = self._generate_G_from_H(H)
        G_new = (1 - self.theta) * G + self.theta * G_new
        self.adj = G  # 保存看看学的结果
        # self.adj = self.vertex_assignment_map.detach().data# 保存看看学的结果
        if self.only_G:
            return G_new  # torch.spmm(G, feats)  # nearest_feature

        return torch.spmm(G, feats)  # nearest_feature


if __name__ == '__main__':
    import numpy as np

    dist = np.random(5, 5)
    x = torch.randn(5, 7)
    conv = Flgc2d(10, 6, dist)
    out = conv(x, 2)
    print(out)
