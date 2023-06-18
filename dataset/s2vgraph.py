class S2VGraph(object):
    def __init__(self, g, label, node_features=None, node_tags=None, ):
        """
            g: 邻接矩阵 coo 稀疏格式。 #a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        """
        self.label = label
        self.g = g  # coo-matrix
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0
        self.max_neighbor = 0
