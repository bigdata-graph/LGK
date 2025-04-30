import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_approximation import Nystroem


from util import S2VGraph
from scipy.sparse.linalg import eigs
sys.path.append("models/")
from mlp import MLP
# ########################################我改lagcn
import cvae_pretrain
from Load_dataset import load_split_MUTAG_data
from utils import load_data1, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp
from cvae_models import VAE
from tqdm import tqdm, trange
from models1 import LAGCN
from grakel import GraphKernel, WeisfeilerLehman, VertexHistogram


# ########################################我改lagcn
class GraphCNN(nn.Module):
    # 构建MLP层
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
        '''
            num_layers: number o f layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
            num_layers：神经网络中的层数（包括输入层）默认5
            num_mlp_layers：以mlps为单位的层数（不包括输入层）默认2
            input_dim：输入特征的维度
            hidden_dim：所有图层上隐藏单位的维度默认64
            output_dim：用于预测的类数
            final_dropout：最终线性层上的dropout比率
            learn_eps：如果为True，则学习epsilon以区分中心节点和相邻节点。如果为False，则聚合相邻节点和中心节点。默认true
            neighbor_poolingtype：如何聚合邻居（平均值、平均值或最大值）默认sum
            graph_poolingtype：如何聚合图中的整个节点（平均值、平均值）默认sum
            device：要使用的设备默认gpu
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        ###List of MLPs定义mlp多层感知机
        self.mlps = torch.nn.ModuleList()
        self.mlps1 = torch.nn.ModuleList()
        # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
        # self.mlps2 = torch.nn.ModuleList()
        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        # 应用于MLP输出（最终预测线性层输入）的批规范列表
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms1 = torch.nn.ModuleList()
        # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
        # self.batch_norms2 = torch.nn.ModuleList()
        # 0-4,一共构建了5个MLP
        for layer in range(self.num_layers-1):
            if layer == 0:
                input_dim:4
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
                self.mlps1.append(MLP(num_mlp_layers, 19, hidden_dim, hidden_dim))
                # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
                # self.mlps2.append(MLP(num_mlp_layers, 1000, 4, 4))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                self.mlps1.append(MLP(num_mlp_layers, 19, hidden_dim, hidden_dim))
                # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
                # self.mlps1.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
                # self.mlps2.append(MLP(num_mlp_layers, 1000, 4, 4))
            # 批次内的特征进行归一化
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.batch_norms1.append(nn.BatchNorm1d(hidden_dim))
            # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
            # self.batch_norms2.append(nn.BatchNorm1d(4))
        # 将dofferemt层的隐藏表示映射到预测分数的线性函数
        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        self.linears_prediction1 = torch.nn.ModuleList()
        # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
        # self.linears_prediction2 = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
                self.linears_prediction1.append(nn.Linear(19, output_dim))
                # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
                # self.linears_prediction2.append(nn.Linear(1000, 4))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))
                self.linears_prediction1.append(nn.Linear(hidden_dim, output_dim))
                # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
                # self.linears_prediction2.append(nn.Linear(4, 4))

    # 返回邻居节点
    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph
        # 在连接图中创建padded_neighbor_list

        # 计算当前小型批处理中图形内的最大邻居数
        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                # 将偏移值添加到相邻索引
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                # 填充，假设虚拟数据存储在-1中
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                # 如果learn_eps为False，则在最大池中添加中心节点，即聚合中心节点和相邻节点。
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    # 创建块对角稀疏矩阵,邻接矩阵
    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix创建块对角稀疏矩阵

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            # 不同图的索引
            start_idx.append(start_idx[i] + len(graph.g))
            # print("graph.edge_mat",graph.edge_mat)
            edge_mat_list.append(graph.edge_mat + start_idx[i])
            # print("start_idx",start_idx)
            # print("edge_mat_list", edge_mat_list)
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
        # 如果learn_eps为False，则在邻接矩阵中添加自循环，即聚合中心节点和相邻节点。
        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)
        # print("Adj_block", np.array(Adj_block_idx).shape)
        # print("Adj_block", np.array(Adj_block_elem).shape)
        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))
        # print("Adj_block",Adj_block)
        return Adj_block.to(self.device)

    # # 8888888888888888888888888888888888888888888888888888888888888888获取每个图的邻接稀疏矩阵，用于进行WL图核计算
    # def __adj(self, batch_graph):
    #     ###create block diagonal sparse matrix创建块对角稀疏矩阵
    #     edge_mat_list = []
    #     start_idx = [0]
    #     graph = batch_graph
    #     start_idx.append(start_idx[0] + len(graph.g))
    #     edge_mat_list.append(graph.edge_mat + start_idx[0])
    #     Adj_block_idx = torch.cat(edge_mat_list, 1)
    #     Adj_block_elem = torch.ones(Adj_block_idx.shape[1])
    # 
    #     #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
    #     # 如果learn_eps为False，则在邻接矩阵中添加自循环，即聚合中心节点和相邻节点。
    #     if not self.learn_eps:
    #         num_node = start_idx[-1]
    #         self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
    #         elem = torch.ones(num_node)
    #         Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
    #         Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)
    # 
    #     Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))
    #     # print("Adj_block",Adj_block)
    #     return Adj_block.to(self.device)
    # # 8888888888888888888888888888888888888888888888888888888888888888
    # 在每个图中的整个节点上创建总和或平均池稀疏矩阵（num个图x num个节点）
    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)

        start_idx = [0]

        #compute the padded neighbor list计算邻居数
        for i, graph in enumerate(batch_graph):
            # print("len(graph.g)",len(graph.g))
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling平均池
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling求和池
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling元素最小值永远不会影响最大池

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 
        # 通过epsilon重加权分别合并相邻节点和中心节点。
        if self.neighbor_pooling_type == "max":
            ##If max pooling最大池化
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling平均池化或者总和池化
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  合并相邻节点和中心节点
            
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            # print("Adj_block",Adj_block.shape)##Adj_block torch.Size([562, 562])
            # print("h", h.shape)##h torch.Size([562, 64])
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes 相邻节点和中心节点的表示
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        # non-linearity
        h = F.relu(h)
        return h
    def next_layer_K(self, selected_rows_pca, layer):
        pooled_rep = self.mlps1[layer](selected_rows_pca.float().to(self.device))
        selected_rows_pca = self.batch_norms1[layer](pooled_rep)
        # non-linearity
        selected_rows = F.relu(selected_rows_pca)
        return selected_rows_pca
    def next_layer_K1(self, selected_rows_pca, layer):
        pooled_rep = self.mlps2[layer](selected_rows_pca.float().to(self.device))
        selected_rows_pca = self.batch_norms2[layer](pooled_rep)
        # non-linearity
        selected_rows = F.relu(selected_rows_pca)
        return selected_rows_pca
    i = 0
    def forward(self,batch_graph, selected_rows):
        # 获取图的邻接矩阵
        # Adj_block_0 = adj1.to_dense().to(torch.float32).cpu()
        # print("Adj_block_0", Adj_block_0)
        # values_tensor = Adj_block_0  # 填充完整的一维张量数据
        # size = (len(Adj_block_0), len(Adj_block_0))  # 邻接矩阵的维度
        # # 创建零矩阵
        # adj_matrix0 = torch.zeros(size)
        # # 将一维张量中的值填入邻接矩阵
        # adj_matrix0.view(-1)[values_tensor.nonzero()] = 1
        # print("adj_matrix0",adj_matrix0)
        # # print("Adj_blockAdj_block", Adj_block[1])
        # # print("Adj_blockAdj_block", Adj_block[1].to_dense())
        # Adj_block_1 = adj2.to_dense().to(torch.float32).cpu()
        # values_tensor1 = Adj_block_1  # 填充完整的一维张量数据
        # size = (len(Adj_block_1), len(Adj_block_1))  # 邻接矩阵的维度
        # # 创建零矩阵
        # adj_matrix1 = torch.zeros(size)
        # # 将一维张量中的值填入邻接矩阵
        # adj_matrix1.view(-1)[values_tensor1.nonzero()] = 1
        # # print("adj_matrix1", adj_matrix1)
        # adj_matrix_0 = adj_matrix0.cpu().numpy().astype(np.int32)
        # adj_matrix_1 = adj_matrix1.cpu().numpy().astype(np.int32)
        # # adj_matrix_0 = np.array(adj_matrix_0)
        # print("adj_matrix_0", adj_matrix_0)
        # print("adj_matrix_1", adj_matrix_1)
        # for i in range(0,len(graphs)):
        #     for j in range(0,len(graphs)):
        #         adj1 = self.__adj(graphs[0])
        #         adj2 = self.__adj(graphs[1])
        #         adj_matrix_1 = adj1.to_dense()
        #         adj_matrix_2 = adj2.to_dense()
        #         adj_matrix_1 = adj_matrix_1.cpu().numpy().astype(np.int32)
        #         adj_matrix_2 = adj_matrix_2.cpu().numpy().astype(np.int32)
        #         print("adj_matrix_1",adj_matrix_1)
        #         # 创建一个无向图对象
        #         graph1 = nx.Graph()
        #         graph2 = nx.Graph()
        #         # 根据邻接矩阵添加节点
        #         num_nodes1 = adj_matrix_1.shape[0]
        #         for i in range(num_nodes1):
        #             graph1.add_node(i)
        #
        #         # 根据邻接矩阵添加边
        #         for i in range(num_nodes1):
        #             for j in range(i + 1, num_nodes1):
        #                 if adj_matrix_1[i, j] == 1:
        #                     graph1.add_edge(i, j)
        #
        #         num_nodes2 = adj_matrix_2.shape[0]
        #         for i in range(num_nodes2):
        #             graph2.add_node(i)
        #
        #         # 根据邻接矩阵添加边
        #         for i in range(num_nodes2):
        #             for j in range(i + 1, num_nodes2):
        #                 if adj_matrix_2[i, j] == 1:
        #                     graph2.add_edge(i, j)
        #         print(graph1)
        #         gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
        #         K_train = gk.fit_transform([graph1,graph2], output_trainresult=True)
        #         print("K_train",K_train)
                # def weisfeiler_lehman_kernel(adj_matrix1, adj_matrix2, iterations):
                #     # Initialize labels for nodes in both graphs
                #     num_nodes = adj_matrix1.shape[0]
                #     num_nodes2 = adj_matrix2.shape[0]
                #     labels1 = np.arange(num_nodes)
                #     labels2 = np.arange(num_nodes2)
                #     for _ in range(iterations):
                #         labels1_new = np.zeros(num_nodes, dtype=int)
                #         labels2_new = np.zeros(num_nodes, dtype=int)
                #         # Update labels for graph 1
                #         for node in range(num_nodes):
                #             neighbors = np.where(adj_matrix1[node] == 1)[0]
                #             neighbor_labels = labels1[neighbors]
                #             label = tuple(sorted(np.concatenate(([labels1[node]], neighbor_labels))))
                #             label_hash = hash(label)  # Limit hash to a smaller range
                #             labels1_new[node] = label_hash
                #         # Update labels for graph 2
                #         for node in range(num_nodes2):
                #             neighbors = np.where(adj_matrix2[node] == 1)[0]
                #             neighbor_labels = labels2[neighbors]
                #             label = tuple(sorted(np.concatenate(([labels2[node]], neighbor_labels))))
                #             label_hash = hash(label)  # Limit hash to a smaller range
                #             labels2_new[node] = label_hash
                #         labels1 = labels1_new
                #         labels2 = labels2_new
                #     # Calculate similarity by counting matching labels
                #     similarity = np.sum(labels1 == labels2) / num_nodes
                #     return similarity
                #
                # iterations = 3
                # similarity = weisfeiler_lehman_kernel(adj_matrix_0, adj_matrix_1, iterations)
                # print("Graph Similarity:", similarity)


                # g0 = nx.from_numpy_matrix(adj_matrix_0)
                # g1 = nx.from_numpy_matrix(adj_matrix_1)
                # node_labels_g0 = list(range(len(adj_matrix_0)))
                # node_labels_g1 = list(range(len(adj_matrix_1)))

                # Convert adjacency matrices to graph objects
                # graph1 = {"adjacency": adj_matrix_0}
                # graph2 = {"adjacency": adj_matrix_1}
                #
                # # Initialize Weisfeiler-Lehman graph kernel
                # wl_kernel = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 3}])
                # graphs = [graph1, graph2]
                # # Calculate kernel matrix
                # kernel_matrix = wl_kernel.fit_transform(graphs)
                #
                # print("Kernel Matrix:")
                # print(kernel_matrix)
                # g0_grakel = Graph(adj_matrix_0)
                # g1_grakel = Graph(adj_matrix_1)
                # nx.set_node_attributes(g0_grakel, {node: label for node, label in zip(g0_grakel.nodes, node_labels_g0)},
                #                        name='label')
                # nx.set_node_attributes(g1_grakel, {node: label for node, label in zip(g1_grakel.nodes, node_labels_g1)},
                #                        name='label')
                # wl_kernel = WeisfeilerLehman(n_iter=5)
                # kernel_matrix = wl_kernel.fit_transform([g0_grakel, g1_grakel], output_trainresult=True)
                # print("kernel_value", kernel_matrix[0, 1])

        # 图的节点特征
        X_listtotal = []
        # print("selected_rows",selected_rows.shape)
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        # print("node_features",batch_graph[0])
        # print("batch_graph[0].node_features",batch_graph[10].node_features)
        # print("X_concat",isinstance(X_concat,torch.Tensor))
        # *******************************************lgacn
        for i in range(0,len(batch_graph)):
            def get_augmented_features(concat):
                # cvae_features = torch.tensor(batch_graph[i].node_features, dtype=torch.float32)
                cvae_features = batch_graph[i].node_features.clone().detach().float()
                cvae = VAE(encoder_layer_sizes=[batch_graph[i].node_features.shape[1], 256],
                           latent_size=8,
                           decoder_layer_sizes=[256, batch_graph[i].node_features.shape[1]],
                           conditional=8,
                           conditional_size=batch_graph[i].node_features.shape[1])

                z = torch.randn([cvae_features.size(0), cvae.latent_size])
                augmented_features = cvae.inference(z, cvae_features)
                augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
                return augmented_features

            X_list = get_augmented_features(1)
            # print("X_listX_list",X_list)
            X_listtotal.append(X_list)
            # print("len(X_listtotal)", len(X_listtotal))
            X_listcat = X_listtotal[0].cpu().numpy()
            for i in range(1,len(X_listtotal)):
                X_listcat = np.vstack((X_listcat,X_listtotal[i].cpu().numpy()))
            # print("X_listcat",X_listcat.shape)
        X_listcat=torch.tensor(X_listcat).to(self.device)
        # print("X_listcat",X_listcat)
        # *******************************************lgacn

        # 在每个图中的整个节点上创建总和或平均池稀疏矩阵（num个图x num个节点）
        graph_pool = self.__preprocess_graphpool(batch_graph)
        # 获取邻接矩阵
        # print("graph_pool",graph_pool)
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)
            # print("Adj_block", Adj_block)

        #list of hidden representation at each layer (including input)每个层的隐藏表示列表（包括输入）
        hidden_rep = [X_concat]
        # 节点特征表示
        # # *******************************************lgacn
        h = X_concat/2+X_listcat

        num_cols = h.shape[1]
        cols_to_zero = int(0.2 * num_cols)
        zero_cols = np.random.choice(num_cols, cols_to_zero, replace=False)
        # h_augument2[:, zero_cols] = 0
        h[:, zero_cols] = 0
        # h = X_concat
        # h = X_concat + X_listcat
        # 初始化PCA模型，选择降维后的维度数

        # pca = PCA(n_components=19)
        # kpca = KernelPCA(n_components=19, kernel='rbf')
        nystroem = Nystroem(n_components=19)
        # kernel_cea = KernelPCA(n_components=3)
        # 对数据进行PCA降维
        selected_rows_pca = nystroem.fit_transform(selected_rows)

        # # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
        # for layers in range(self.num_layers - 1):
        #     selected_rows_torch = torch.tensor(selected_rows)
        #     selected_rows_torch = self.next_layer_K1(selected_rows_torch, layers)
        # # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
        # print("selected_rows_torch",selected_rows_torch.shape)
        # print("selected_rows_torch",np.array(selected_rows_torch).shape)
        # selected_rows_pca = kpca.fit_transform(selected_rows)
        # KCEA
        # selected_rows_pca = kernel_cea.fit_transform(selected_rows)
        # kernel_cea = KernelPCA(n_components=19)
        # nystroem = Nystroem(n_components=4)
        # selected_rows_pca = kernel_cea.fit_transform(selected_rows)
        # selected_rows_pca = selected_rows
        # selected_rows_pca = nystroem.fit_transform(selected_rows)
        # print("selected_rows_pca",selected_rows_pca.shape)
        # # *******************************************lgacn
        # h = X_concat
        # print("hhhhhhhhhhhhhhhhhh",h.shape) ## hhhhhhhhhhhhhhhhhh torch.Size([576, 7])
        # 通过不同池化方式池化节点特征表示\\\\\
        for layer in range(self.num_layers-1):
            # next_layer_eps通过epsilon重加权分别合并相邻节点和中心节点。然后选择具体的池化操作
            # next_layer合并相邻节点和中心节点.然后选择具体的池化操作
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
                # selected_rows = self.next_layer_K(selected_rows, layer)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
                # selected_rows = self.next_layer_K(selected_rows, layer)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
                # selected_rows = self.next_layer_K(selected_rows, layer)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)
                # h1 = self.next_layer(h1, layer, Adj_block=Adj_block)
                selected_rows_torch = torch.tensor(selected_rows_pca)
                # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
                # selected_rows_torch = torch.tensor(selected_rows_torch)
                # MLP降维# MLP降维# MLP降维# MLP降维# MLP降维
                selected_rows_torch = self.next_layer_K(selected_rows_torch, layer)
            # print("h",h.shape)# # 562*64
            # print("selected_rows_torch",selected_rows_torch.shape)##selected_rows_torch torch.Size([32, 64])
            # h = h+h1
            hidden_rep.append(h)
        # print("hhhhhhhhhhhhhhhhhh", hidden_rep[0].shape)
        score_over_layer = 0
        list1=[]
        #perform pooling over all nodes in each graph in every layer对每个层中每个图中的所有节点执行池化

        # num_cols = h.shape[1]
        # cols_to_zero = int(0.8 * num_cols)  # 20% 的列数
        #
        # # 随机选择要置零的列索引
        # zero_cols = np.random.choice(num_cols, cols_to_zero, replace=False)
        #
        # # 将选定列的值设置为 0
        # h[:, zero_cols] = 0
        # # print("h",h)

        # # 计算20%行数对应的数量
        # num_rows = h.shape[0]
        # rows_to_change = int(0.8 * num_rows)
        #
        # # 随机选择要更改的行索引
        # rows_indices = np.random.choice(num_rows, rows_to_change, replace=False)
        #
        # # 将选定的行的值设置为0
        # h[rows_indices] = 0

        for layer, h in enumerate(hidden_rep):
            # 矩阵乘法

            if(layer == 0):
                # print("h",h.shape)# # 562*64

                pooled_h = torch.spmm(graph_pool, h)
                # print("graph_poolgraph_pool",graph_pool.shape)# # 32*562/64,64,41*562
                # pooled_k = torch.spmm(graph_pool, selected_rows)
                # print("pooled_h",pooled_h.shape)
                # # ******************************juli
                # # 32*64/64,64,41*64
                # print("pooled_hpooled_h",pooled_h.shape)
                # # print("pooled_hpooled_hpooled_hpooled_h", pooled_h)
                # # 输出分数
                # list1.append(pooled_h)
                # i_jsum = []
                # i_jtotal = []
                # for i in range(0, len(pooled_h)):
                #     for j in range(i, len(pooled_h)):
                #         i_j = (pooled_h[i] - pooled_h[j]).detach()
                #         xy = np.exp(-((i_j / 100))) / 2
                #         # xy = ((np.exp(-(i_j/100)))*(1+(i_j)/100))/4
                #         if (j == 1):
                #             i_jsum = xy
                #         else:
                #             i_jsum += xy
                #     i_jtotal.append(i_jsum)
                # # print(i_jtotal)
                # pooled_h+=torch.stack(i_jtotal)
                # # print(pooled_h.shape)
                # # ******************************juli
                score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout,
                                              training=self.training)
            else:
                # print("h",h.shape)# # 562*64
                pooled_h = torch.spmm(graph_pool, h)
                pooled_h = pooled_h + selected_rows_torch
                score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout,
                                              training=self.training)

        # print("pooled_hpooled_h",pooled_h.shape)
        return score_over_layer

