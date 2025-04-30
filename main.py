from __future__ import print_function

# from collections import defaultdict
#
# import networkx as nx
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils.tests.test_pprint import GridSearchCV
import datetime
import math
import time

from models import graphcnn

print(__doc__)
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN
# from torch_geometric.datasets import TUDataset
# ########################################我改

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram
# ########################################我改lagcn
import cvae_pretrain
from Load_dataset import load_split_MUTAG_data
from utils import load_data1, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp
from cvae_models import VAE
from tqdm import tqdm, trange
from models1 import LAGCN
from sklearn.decomposition import PCA, KernelPCA
# ########################################我改lagcn


criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch,K):
   model.train()
   total_iters = args.iters_per_epoch
   pbar = tqdm(range(total_iters), unit='batch')
   loss_accum = 0
   # kernel_cea = KernelPCA(n_components=19)
   # selected_rows_pca = kernel_cea.fit_transform(K)
   for pos in pbar:

       selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
       # selected_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
       # print("selected_idx",selected_idx)
       batch_graph = [train_graphs[idx] for idx in selected_idx]
       selected_rows = K[selected_idx, :]
       # selected_rows = selected_rows_pca[selected_idx, :]
       # print("selected_rows",selected_rows.shape)
       # output = model(batch_graph)
       output = model(batch_graph,selected_rows)
       # print("outputoutputoutput", output.shape)
       labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
       # print("labelslabelslabels", labels.shape)
       # compute loss
       loss = criterion(output, labels)
       # backprop
       if optimizer is not None:
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

       loss = loss.detach().cpu().numpy()
       loss_accum += loss

       # report
       pbar.set_description('epoch: %d' % (epoch))

   average_loss = loss_accum / total_iters
   print("loss training: %f" % (average_loss))

   return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
# 在测试过程中，将数据传递给带有小批量的模型，以避免内存溢出（不执行反向传播）
def pass_data_iteratively(model, graphs, K,minibatch_size = 64):
   model.eval()
   output = []
   idx = np.arange(len(graphs))
   for i in range(0, len(graphs), minibatch_size):
       sampled_idx = idx[i:i+minibatch_size]
       # print("sampled_idx",sampled_idx)
       selected_rows = K[sampled_idx, :]
       if len(sampled_idx) == 0:
           continue
       # 对小批量图计算结果
       # score_over_layer = model([graphs[j] for j in sampled_idx]).detach()

       score_over_layer=model([graphs[j] for j in sampled_idx], selected_rows).detach()
       output.append(score_over_layer.detach())
   return torch.cat(output, 0)


def test(args, model, device, train_graphs, test_graphs, epoch,K):
   model.eval()
   output_train= pass_data_iteratively(model, train_graphs ,K)
   # output.shape,169*2
   # print("output************", output_train.shape)
   # print("output************", output_train)
   # 取结果中的最大值作为预测结果
   pred_train = output_train.max(1, keepdim=True)[1]
   # pred_train.shape169*1
   # print("pred************", pred_train.shape)
   labels_train = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
   # print("labels.shape************", labels_train.shape)
   correct = pred_train.eq(labels_train.view_as(pred_train)).sum().cpu().item()
   # 训练精确度
   acc_train = correct / float(len(train_graphs))
# ?????????????????????????????
#     output_test ,list1_4= pass_data_iteratively(model, test_graphs)
   output_test = pass_data_iteratively(model, test_graphs, K )
# ?????????????????????????????0.947368/0.8947368/0.8947368/0.947368/1/0.947368/0.947368/0.947368/0.833333/0.888889

   # print("111111111************", output_test.shape)19*2
   pred_test = output_test.max(1, keepdim=True)[1]
   # print("111111111************", pred_test.shape)19*1
   labels_test = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
   correct = pred_test.eq(labels_test.view_as(pred_test)).sum().cpu().item()
   # 测试精确度
   acc_test = correct / float(len(test_graphs))
   print("accuracy train: %f test: %f" % (acc_train, acc_test))
   # ########################################我改
   def transpose(M):
       # 初始化转置后的矩阵
       result = []
       # 获取转置前的行和列
       row, col = np.shape(M)
       # 先对列进行循环
       for i in range(col):
           # 外层循环的容器
           item = []
           # 在列循环的内部进行行的循环
           for index in range(row):
               item.append(M[index][i])
           result.append(item)
       return result
   output_train = output_train.cpu()
   pred_train = pred_train.cpu()
   output_test = output_test.cpu()
   pred_test = pred_test.cpu()
   labels_train = labels_train.cpu()
   labels_test = labels_test.cpu()
   output = np.vstack((output_train, output_test))
   # print("111111111************", output.shape)
   # output 188*2
   pred = np.vstack((pred_train,pred_test))
   # pred 188*2
   labels = np.hstack((labels_train,labels_test))
   # labels 188*1
   # print("labels************", labels)
   outputT = np.array(transpose(output))
   predT = np.array(transpose(pred))
   # output_trainresult 188*188
   output_result = np.dot(output, outputT)
   output_traintest = np.dot(output_test , transpose(output_train))
   output_traintest_mean = np.mean(output_traintest)
   # print("output_traintest_mean",output_traintest_mean)
   # print("output_result",output_result)
   pred_result = np.dot(pred, predT)
   output_trainT = np.array(transpose(output_train))
   output_trainresult = np.dot(output_train, output_trainT)
   # output_trainresult.shape169*169,output_testresult.shape19*169
   # print("output_trainresultoutput_trainresult", output_trainresult.shape)
   output_testresult = np.dot(output_test, output_trainT)
   # print("output_testresultoutput_testresult", output_testresult.shape)

   # output1 = output_train
   # output2 = output_train
   # output_train_empty_matrix = np.empty((len(output_train), len(output_train)))
   # for i in range(0, len(output_train)):
   #     for j in range(i, len(output_train)):
   #         i_j = np.array(output1[i] - output2[j])
   #         i_jreshape = i_j.reshape((2, 1))
   #         # print("i_jreshape",i_jreshape.shape)
   #         i_jT = np.transpose(i_jreshape)
   #         # print("i_jT",i_jT.shape)
   #         i_j1 = np.dot(i_jT, i_jreshape)
   #         x_y = i_j1
   #         # print("i_j1",i_j1.shape)
   #         i_j = np.sqrt(x_y)
   #         # print("i_ji_j",i_j)
   #         # 0.2,0.3,0.4
   #         x_y = np.exp(-((i_j / 100))) / 2
   #         # print("x_y1",x_y)
   #         # x_y = np.exp(-i_j)
   #         # 0.2
   #         # x_y = ((np.exp(-(i_j / 100))) * (1 + (i_j) / 100)) / 4
   #         # print("x_y2", x_y)
   #         # 0.1
   #         # x_y = ((np.exp(-(i_j / 100))) * (3 + 3 * (i_j) / 100 + (i_j) * (i_j) / 10000)) / 16
   #         # print("x_y3",x_y)
   #         output_train_empty_matrix[i][j] = x_y
   # # print("output_train_empty_matrix",output_train_empty_matrix)
   # # *******************************8
   # # output_train_empty_matrix_x = output_train_empty_matrix*x_y
   # # # output_train_empty_matrix_x_x = output_train_empty_matrix * x_y * x_y
   # # output_train_empty_matrix = (output_train_empty_matrix + output_train_empty_matrix_x)/2
   # # output_train_empty_matrix = (3*(output_train_empty_matrix + output_train_empty_matrix_x + output_train_empty_matrix_x_x)) / 8
   # # print("output_train_empty_matrix1", output_train_empty_matrix)
   # # *********************************8
   # scaler = MinMaxScaler()
   # scaler_data = scaler.fit_transform(output_train_empty_matrix)
   # # print("output_train_empty_matrix",output_train_empty_matrix)
   # output_trainresult = scaler_data
   # output_trainresult = output_train_empty_matrix
   # # missing_values = np.isnan(output_trainresult)
   # # output_trainresult = np.where(missing_values, 0 ,output_trainresult)
   # # inf_values = np.isinf(output_trainresult)
   # # output_trainresult = np.where(inf_values, 0, output_trainresult)
   # output_trainresult = np.nan_to_num(output_trainresult.astype(np.float64))
   # # print("output_trainresult",output_trainresult)
   # # print(inf_values)
   # # output_trainresult = output_train_empty_matrix
   # output3 = output_test
   # output4 = output_train
   # output_test_empty_matrix = np.empty((len(output_test), len(output_train)))
   # for i in range(0, len(output_test)):
   #     for j in range(i, len(output_train)):
   #         i_j = np.array(output3[i] - output4[j])
   #         i_jreshape = i_j.reshape((2, 1))
   #         # print("i_jreshape",i_jreshape.shape)
   #         i_jT = np.transpose(i_jreshape)
   #         # print("i_jT",i_jT.shape)
   #         i_j1 = np.dot(i_jT, i_jreshape)
   #         x_y = i_j1
   #         # print("i_j1",i_j1.shape)
   #         i_j = np.sqrt(x_y)
   #         # print("i_ji_j",i_j)
   #         x_y = np.exp(-((i_j / 100))) / 2
   #         # x_y = ((np.exp(-(i_j / 100))) * (3 + 3 * (i_j) / 100 + (i_j) * (i_j) / 10000)) / 16
   #         # x_y = ((np.exp(-(i_j/100)))*(1+(i_j)/100))/4
   #         # x_y = ((np.exp(-(i_j / 100))) * (1 + (i_j) / 100)) / 3
   #         # print("x_y",x_y)
   #         output_test_empty_matrix[i][j] = x_y
   #
   # # print("output_test_empty_matrix", output_test_empty_matrix)
   # output_testresult = output_test_empty_matrix
   # #
   # # # ********************************
   # # output_test_empty_matrix_mean = np.mean(output_test_empty_matrix)
   # # # print("output_test_empty_matrix_mean",output_test_empty_matrix_mean)
   # # mean = output_traintest_mean-output_test_empty_matrix_mean
   # # # print("mean",output_traintest_mean-output_test_empty_matrix_mean)
   # # output_test_empty_matrix_x = output_test_empty_matrix * x_y
   # # # output_test_empty_matrix_x_x = output_test_empty_matrix * x_y * x_y
   # # # # print("output_test_empty_matrix_x",output_test_empty_matrix_x)
   # # output_test_empty_matrix = (output_test_empty_matrix + output_test_empty_matrix_x) / 2
   # # # output_testresult = (3*(output_test_empty_matrix + output_test_empty_matrix_x +output_test_empty_matrix_x_x))/8
   # # # output_test_empty_matrix1 = output_testresult - output_test_empty_matrix
   # # # print("output_test_empty_matrix1", output_test_empty_matrix1)
   # # output_testresult = output_test_empty_matrix + mean
   # # output_testresult = np.where(output_test_empty_matrix != 0, output_test_empty_matrix + mean, output_test_empty_matrix)
   # # output_testresult = np.where(output_testresult != 0, output_testresult + 0.1,
   # # #                              output_testresult)
   # # print("output_testresult",output_testresult)
   # # output_test_empty_matrix1 = output_testresult - output_traintest
   # # print("output_test_empty_matrix1", output_test_empty_matrix1)
   # # missing_values = np.isnan(output_testresult)
   # # output_testresult = np.where(missing_values, 0, output_testresult)
   # # inf_values = np.isinf(output_testresult)
   # # output_testresult = np.where(inf_values, 0, output_testresult)
   # output_testresult = np.nan_to_num(output_testresult.astype(np.float64))
   # scaler = MinMaxScaler()
   # scaler_data1 = scaler.fit_transform(output_testresult)
   # # print("output_testresult",scaler_data1)
   # output_testresult = scaler_data1
   # # # *******************88*******
   # #
   # # output_trainT = np.array(transpose(output_train))
   # # # output_trainresult = np.dot(output_train, output_trainT)
   # # # output_trainresult.shape169*169,output_testresult.shape19*169
   # # # print("output_trainresultoutput_trainresult",output_trainresult.shape)
   # # output_testresult = np.dot(output_test, output_trainT)
   # # print("output_testresultoutput_testresult", output_testresult.shape)
   # # missing_values1 = np.isnan(output_testresult)
   # # output_testresult = np.where(missing_values1, 0, output_testresult)
   # # inf_values1 = np.isinf(output_testresult)
   # # output_testresult = np.where(inf_values1, 0, output_testresult)
   # # output_testresult = np.nan_to_num(output_testresult.astype(np.float64))
   # # scaler = MinMaxScaler()
   # # scaler_data2 = scaler.fit_transform(output_testresult)
   # # # print("output_testresult",scaler_data1)
   # # output_testresult = scaler_data2
   # # print("output_testresult",output_testresult)

   # 训练核169*169
   # 测试核19*169
   def SVC_classify1(output_trainresult, output_testresult, labels_train, labels_test, y, search):
       accuracies1 = []
       # # x_train, x_test = x[0:900, 0:900], x[900:1000, 0:900]
       # x_train, x_test = x[0:900, ], x[900:1000, ]
       # print("x_trainx_trainx_train",x_train.shape)
       # print("x_testx_testx_test", x_test.shape)
       y_train, y_test = y[0:len(labels_train)], y[len(labels_train):344]
       # print("y_trainy_trainy_train",y_train.shape)
       # print("y_testy_testy_test",y_test.shape)
       if search:
           params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
           # estimator：选择使用的分类器,这里是SVC()，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法
           # cv = None：交叉验证参数，默认None，使用五折交叉验证。指定fold数量，默认为5(之前版本为3)，也可以是yield训练/测试数据的生成器。
           # ’linear’：线性核函数‘poly’：多项式核函数‘rbf’：径像核函数 / 高斯核‘sigmod’：sigmod核函数
           # ‘precomputed’：核矩阵，precomputed表示自己提前计算好核函数矩阵
           classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
       else:
           classifier = SVC(C=10)
       classifier.fit(output_trainresult, y_train)

       accuracies1.append(accuracy_score(y_test, classifier.predict(output_testresult)))
       ac = accuracy_score(y_test, classifier.predict(output_testresult))
       missing_values = np.isnan(ac)
       ac = np.where(missing_values, 0, ac)
       inf_values = np.isinf(ac)
       ac = np.where(inf_values, 0, ac)
       ac = np.nan_to_num(ac.astype(np.float64))
       # print("ac",ac)
       accuracies1.append(ac)
       accuracies1 = np.array(accuracies1)
       return accuracies1.mean()

   accuracies1 = SVC_classify1(output_trainresult, output_testresult, labels_train, labels_test, labels, 1)
   print("accuracies1accuracies1accuracies1accuracies1accuracies1accuracies1", accuracies1)

   # 训练核169*188
   # 测试核16*188
   # def SVC_classify(x, y, search):
   #     accuracies = []
   #     # x_train, x_test = x[0:900, 0:900], x[900:1000, 0:900]
   #     x_train, x_test = x[0:169, ], x[169:188, ]
   #     print("x_trainx_trainx_train",x_train.shape)
   #     print("x_testx_testx_test", x_test.shape)
   #     y_train, y_test = y[0:169], y[169:188]
   #     # print("y_trainy_trainy_train",y_train.shape)
   #     # print("y_testy_testy_test",y_test.shape)
   #     if search:
   #         params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
   #         # estimator：选择使用的分类器,这里是SVC()，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法
   #         # cv = None：交叉验证参数，默认None，使用五折交叉验证。指定fold数量，默认为5(之前版本为3)，也可以是yield训练/测试数据的生成器。
   #         # ’linear’：线性核函数‘poly’：多项式核函数‘rbf’：径像核函数 / 高斯核‘sigmod’：sigmod核函数
   #         # ‘precomputed’：核矩阵，precomputed表示自己提前计算好核函数矩阵
   #         classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
   #     else:
   #         classifier = SVC(C=10)
   #     classifier.fit(x_train, y_train)
   #
   #     accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
   #     accuracies = np.array(accuracies)
   #     return accuracies.mean()
   #
   # accuracies = SVC_classify(output_result, labels, 1)
   # print("accuraciesaccuraciesaccuraciesaccuraciesaccuraciesaccuracies", accuracies)
   # # ########################################我改2
   # # Loads the MUTAG dataset
   # MUTAG = fetch_dataset("MUTAG", verbose=False)
   # G, y = MUTAG.data, MUTAG.target
   # # G是188个图的数据，数据中包含边连接矩阵，节点标签，边标签。
   # # y是188个图对应的标签labels（188，1），labels是判断化合物是芳香族还是杂芳族所以只有1/-1。
   # # Splits the dataset into a training and a test set
   # G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)
   #
   # # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
   # # 利用wl子树核创造核矩阵gk
   # gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
   # # print("gk**********)", gk)
   # # 在同一数据集上进行拟合和变换。
   # # print("G_train*****************",G_train)
   # K_train = gk.fit_transform(G_train, output_trainresult)
   # K_test = gk.transform(G_test)
   # # K_train.shape(169, 169),K_test.shape(19, 169)
   # # print("K_train*****************", K_train)
   # # print("K_test*****************", K_test)
   # # print("K_test.shape*****************", K_test.shape)
   # # Uses the SVM classifier to perform classification
   # # 利用SVM分类器执行分类
   # clf = SVC(kernel="precomputed")
   # def normalization(output_trainresult):
   #     X_diag = np.diagonal(output_trainresult)
   #     old_settings = np.seterr(divide='ignore')
   #     # km = np.nan_to_num(np.divide(km, np.sqrt(np.outer(self._X_diag, self._X_diag))))
   #     km = np.nan_to_num(np.divide(output_trainresult, np.sqrt(np.outer(X_diag, X_diag))))
   #     np.seterr(**old_settings)
   #     print(km.shape)
   #     return km
   # output_trainresult = normalization(output_trainresult)
   # # print("output_trainresult******************", output_trainresult)
   # clf.fit(output_trainresult, y_train)
   # y_1 = clf.predict(output_trainresult)
   #
   # print("clf.fit(K_train, y_train)", clf.predict(output_trainresult))
   # acc = accuracy_score(y_train, y_1)
   # print("Accuracy1:", str(round(acc * 100, 2)) + "%")
   # # ########################################我改
   # # ########################################我改
   #
   #
   # # ########################################我改
   # output_test = output.cpu()
   # pred_test = pred.cpu()
   # outputtest = np.array(transpose(output_test))
   # # output_testresult = np.dot(output_test, outputtest)
   # pred_testresult = np.dot(output_test, outputtest)
   # # print("pred_trainresult%%%%%%%%%%%%%%%",pred_trainresult)
   # pred_testresult = normalization(pred_testresult)
   # clf.fit(pred_testresult, y_test)
   # y_2 = clf.predict(pred_testresult)
   # # y_pred = clf.predict(pred_trainresult)
   # print("y_test%%%%%%%%%%%%",y_test)
   # print("clf.fit(K_train, y_train)", clf.predict(pred_testresult))
   # # Computes and prints the classification accuracy计算分类精度
   # acc = accuracy_score(y_test, y_2)
   # print("Accuracy:", str(round(acc * 100, 2)) + "%")

   # 将结果转置
   # outputT = np.array(transpose(output))
   # print("outputT.shape*******************", outputT.shape)
   # output_result = np.dot(output,outputT)
   # print("output_result.shape*******************", output_result.shape)
   # print("output_result*******************", output_result)
   # ########################################我改

   # return acc_train, acc_test, accuracies, accuracies1
   return acc_train, acc_test, accuracies1
   # model.eval()
   #
   # output = pass_data_iteratively(model, train_graphs)
   # pred = output.max(1, keepdim=True)[1]
   # labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
   # correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
   # acc_train = correct / float(len(train_graphs))
   #
   # output = pass_data_iteratively(model, test_graphs)
   # pred = output.max(1, keepdim=True)[1]
   # labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
   # correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
   # acc_test = correct / float(len(test_graphs))
   #
   # print("accuracy train: %f test: %f" % (acc_train, acc_test))
   #
   # return acc_train, acc_test
def main():
   start = time.time()
   # Training settings
   # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
   parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
   # 默认数据集MUTAG
   parser.add_argument('--dataset', type=str, default="PTC",
                       help='name of dataset (default: IMDBMULTI)')
   # 设备默认gpu
   parser.add_argument('--device', type=int, default=0,
                       help='which gpu to use if any (default: 0)')
   parser.add_argument('--batch_size', type=int, default=32,
                       help='input batch size for training (default: 32)')
   # 每个epoch的迭代次数（默认值：50）
   parser.add_argument('--iters_per_epoch', type=int, default=50,
                       help='number of iterations per each epoch (default: 50)')
   # epoch默认350
   parser.add_argument('--epochs', type=int, default=300,
                       help='number of epochs to train (default: 350)')
   # 学习率
   parser.add_argument('--lr', type=float, default=0.01,
                       help='learning rate (default: 0.01)')
   parser.add_argument('--seed', type=int, default=0,
                       help='random seed for splitting the dataset into 10 (default: 0)')
   # 10倍验证中的倍数指数。应小于10。
   parser.add_argument('--fold_idx', type=int, default=0,
                       help='the index of fold in 10-fold validation. Should be less then 10.')
   # 默认5层
   parser.add_argument('--num_layers', type=int, default=5,
                       help='number of layers INCLUDING the input one (default: 5)')
   # mlp2层
   parser.add_argument('--num_mlp_layers', type=int, default=2,
                       help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
   # 隐藏层64维
   parser.add_argument('--hidden_dim', type=int, default=64,
                       help='number of hidden units (default: 64)')
   parser.add_argument('--final_dropout', type=float, default=0.5,
                       help='final layer dropout (default: 0.5)')
   # 默认sum池化
   parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                       help='Pooling for over nodes in a graph: sum or average')
   # 邻居节点池化默认sum
   parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                       help='Pooling for over neighboring nodes: sum, average or max')
   # 是否学习中心节点的ε加权。但不影响训练准确性。
   parser.add_argument('--learn_eps', action="store_true",
                                       help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
   # 假设输入节点特征是节点的度（未标记图的启发式算法
   parser.add_argument('--degree_as_tag', action="store_true",
                   help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
   parser.add_argument("--concat", type=int, default=1)
   # 输出文件名字
   parser.add_argument('--filename', type = str, default = "",
                                       help='output file')
   args = parser.parse_args()
   # print(args.num_layers)

   # # Loads the MUTAG dataset
   # MUTAG = fetch_dataset("MUTAG", verbose=False)
   # G, y = MUTAG.data, MUTAG.target
   # file_contents = []
   # with open('dataset/MUTAG/10fold_idx/test_idx-1.txt', 'r') as file:
   #      content = file.read().splitlines()
   #      file_contents.append(content)
   # file_contents1 = file_contents[0]
   # # print("file_contents1",file_contents1)
   # int_file_contents = []
   # for char in file_contents1:
   #     int_value = int(char)
   #     int_file_contents.append(int_value)
   # # print("int_file_contents",int_file_contents)
   # # Splits the dataset into a training and a test set
   # # G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)
   # G_train = [value for index, value in enumerate(G) if index not in int_file_contents]
   # print(len(G_train))
   # graphs = G_train
   # # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
   # gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
   # gk.fit(graphs)
   # K_train = gk.transform(graphs)
   # print("K_train",K_train.shape)
   # K_test = gk.transform(G_test)

   #set up seeds and gpu device
   torch.manual_seed(0)
   np.random.seed(0)
   device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(0)
   # 下载数据和分类种类
   graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
   # 训练核测试图分割,十字交叉法
   # dataset = TUDataset(root='',name='ENZYMES')
   # print("num_node_features",dataset[0].x)
   maxaccuracies1 = []
   for i in range(0, 10):
       args.fold_idx = i
       # print("args.fold_idx",args.fold_idx)
       train_graphs, test_graphs ,train_idx, test_idx= separate_data(graphs, args.seed, args.fold_idx)
       # print("graphsgraphsgraphs", train_graphs)
       # Loads the MUTAG dataset获取训练数据核矩阵
       # [{(15, 13), (10, 11), (5, 6), (9, 8), (2, 1), (14, 13), (8, 9), (15, 16), (1, 6), (14, 9), (17, 15), (1, 2), (11, 10), (13, 12), (12, 13), (16, 15), (3, 4), (10, 9), (4, 10), (3, 2), (5, 4), (9, 14), (10, 4), (4, 5), (13, 15), (9, 10), (7, 5), (2, 3), (8, 7), (12, 11), (11, 12), (6, 5), (15, 17), (13, 14), (6, 1), (5, 7), (4, 3), (7, 8)}, {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 1, 16: 2, 17: 2}, {(2, 1): 0, (1, 2): 0, (3, 2): 0, (2, 3): 0, (4, 3): 0, (3, 4): 0, (5, 4): 0, (4, 5): 0, (6, 5): 0, (5, 6): 0, (6, 1): 0, (1, 6): 0, (7, 5): 0, (5, 7): 0, (8, 7): 0, (7, 8): 0, (9, 8): 0, (8, 9): 0, (10, 9): 0, (9, 10): 0, (10, 4): 0, (4, 10): 0, (11, 10): 0, (10, 11): 0, (12, 11): 0, (11, 12): 0, (13, 12): 0, (12, 13): 0, (14, 13): 0, (13, 14): 0, (14, 9): 0, (9, 14): 0, (15, 13): 1, (13, 15): 1, (16, 15): 2, (15, 16): 2, (17, 15): 1, (15, 17): 1}]
       PTC = fetch_dataset("PTC_MR", verbose=False)
       G, y = PTC.data, PTC.target
       # ###################################获取邻接矩阵
       # # 给定的边集合
       # edges = G[0][0]
       #
       # # 收集所有节点
       # nodes = set()
       # for edge in edges:
       #     nodes.add(edge[0])
       #     nodes.add(edge[1])
       #
       # num_nodes = max(nodes)
       #
       # # 初始化邻接矩阵
       # adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
       #
       # # 填充邻接矩阵
       # for edge in edges:
       #     node1, node2 = edge
       #     adj_matrix[node1 - 1][node2 - 1] = 1  # 设置连接关系为1
       #     adj_matrix[node2 - 1][node1 - 1] = 1  # 对称矩阵
       #
       # # # 输出邻接矩阵
       # # for row in adj_matrix:
       # #     print(row)
       # # 假设你有邻接矩阵和节点特征
       # # 创建一个图对象
       # # 创建图
       # G = nx.Graph()
       #
       # # # 添加节点
       # # for i in range(len(adj_matrix)):
       # #     G.add_node(i)
       # print("len(adj_matrix)",len(adj_matrix))
       # # 添加节点和设置节点颜色
       # node_features = [
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [0., 1., 0., 0., 0., 0., 0.],
       #  [1., 0., 0., 0., 0., 0., 0.],
       #  [0., 1., 0., 0., 0., 0., 0.],
       #  [0., 0., 1., 0., 0., 0., 0.],
       #  [0., 0., 1., 0., 0., 0., 0.],
       # ]
       # # 增广图可视化# 增广图可视化# 增广图可视化# 增广图可视化
       # def get_augmented_features(concat):
       #     cvae_features = torch.tensor(node_features, dtype=torch.float32)
       #     cvae = VAE(encoder_layer_sizes=[np.array(node_features).shape[1], 256],
       #                latent_size=8,
       #                decoder_layer_sizes=[256, np.array(node_features).shape[1]],
       #                conditional=8,
       #                conditional_size=np.array(node_features).shape[1])
       #     z = torch.randn([cvae_features.size(0), cvae.latent_size])
       #     augmented_features = cvae.inference(z, cvae_features)
       #     augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
       #     return augmented_features
       #
       # X_list = get_augmented_features(1)
       # print("X_list",X_list*1000)
       # rounded_array = np.round(np.array(X_list*1000)).astype(int)
       # print("rounded_array", rounded_array)
       # selected_columns = rounded_array[:, [0, 3, 6]]
       # print("selected_columns", selected_columns)
       # # 将特征值缩放到 0-255 范围内
       # scaled_features = (selected_columns - selected_columns.min()) / (selected_columns.max() - selected_columns.min())
       # print("scaled_features",scaled_features)
       # # 创建图对象
       # G = nx.Graph()
       # # 添加节点并指定节点颜色
       # num_nodes = scaled_features.shape[0]
       # for i in range(num_nodes):
       #     # 将三个特征值作为 RGB 颜色值
       #     color = tuple(scaled_features[i])
       #     G.add_node(i, color=color)
       # for i in range(len(adj_matrix)):
       #     for j in range(len(adj_matrix[i])):
       #         if adj_matrix[i][j] == 1:
       #             G.add_edge(i, j)
       # # 绘制图形
       # pos = nx.spring_layout(G)
       # node_colors = [G.nodes[i]['color'] for i in G.nodes()]
       # # 创建带有颜色映射的图形
       # plt.figure(figsize=(8, 6))
       # nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500)
       # plt.title('Graph with Node Colors based on RGB Features')
       # plt.savefig('1.png', dpi=1200)
       # plt.show()
       # # 增广图可视化# 增广图可视化# 增广图可视化# 增广图可视化

       # # 原图可视化# 原图可视化# 原图可视化# 原图可视化
       # # # 计算节点特征的平均值
       # # avg_features = np.mean(node_features, axis=1)
       # # # 添加节点并指定节点颜色
       # # for i in range(len(avg_features)):
       # #     G.add_node(i, color=avg_features[i])
       # for i in range(len(node_features)):
       #     G.add_node(i)
       #     color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'orange', 5: 'purple', 6: 'cyan'}  # 设置颜色映射
       #     color_index = node_features[i].index(1.0)  # 找到特征值为1的索引作为颜色映射的键
       #     color = color_map.get(color_index, 'black')  # 如果特征值不在颜色映射中，则为黑色
       #     nx.set_node_attributes(G, {i: color}, 'color')
       # # 添加边
       # for i in range(len(adj_matrix)):
       #     for j in range(len(adj_matrix[i])):
       #         if adj_matrix[i][j] == 1:
       #             G.add_edge(i, j)
       #
       # # 绘制图形，并设置节点颜色属性
       # pos = nx.spring_layout(G)
       # node_colors = nx.get_node_attributes(G, 'color').values()
       # nx.draw(G, pos, with_labels=True, node_color=list(node_colors), node_size=500, font_weight='bold')
       # plt.title('Graph with Node Features and Colored Nodes')
       # plt.savefig('2.png', dpi=1200)
       # plt.show()
       # # 绘制图形
       # pos = nx.spring_layout(G)
       # node_colors = [avg_features[node] for node in G.nodes()]
       # nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Reds, node_size=500)
       # sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(avg_features), vmax=max(avg_features)))
       # sm.set_array([])
       # plt.colorbar(sm)
       # plt.title('Graph with Node Colors based on Features')
       # plt.show()
       ###################################获取邻接矩阵
       # ************社交网络加lables
       # for i in range(0,len(G)):
       #     edges = G[i][0]
       #     # Create a dictionary to store node degrees
       #     node_degrees = defaultdict(int)
       #     # Calculate node degrees
       #     for edge in edges:
       #         node1, node2 = edge
       #         node_degrees[node1] += 1
       #         node_degrees[node2] += 1
       #     # Get the range of nodes
       #     all_nodes = set(node for edge in edges for node in edge)
       #     sorted_all_nodes = sorted(all_nodes)
       #     # Create a dictionary with initial degrees set to 0
       #     sorted_node_degrees = {node: 0 for node in sorted_all_nodes}
       #     # Update degrees with calculated values
       #     for node, degree in node_degrees.items():
       #         sorted_node_degrees[node] = degree
       #     G[i][1] = sorted_node_degrees
       # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
       gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
       gk.fit(G)
       K = gk.transform(G)
       print("K",K.shape)

       # 模型GraphCNN
       model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
                        num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                        args.neighbor_pooling_type, device).to(device)

       # 优化模型定义
       optimizer = optim.Adam(model.parameters(), lr=args.lr)
       # 初始化参数
       scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
       maxaccuracies = []
       epoch = []
       maxacc = 0

       epoch1 = []
       maxacc1 = 0
       # avg_loss1 = train1(args, model, device, graphs, optimizer)
       # print("avg_loss1avg_loss1",avg_loss1)
       for epoch in range(1, args.epochs + 1):
       # for epoch in range(1, args.epoch+1):

           # 训练平均损失
           avg_loss = train(args, model, device, train_graphs, optimizer, epoch, K)
           scheduler.step()
           # acc_train, acc_test, accuracies, accuracies1 = test(args, model, device, train_graphs, test_graphs, epoch)
           acc_train, acc_test, accuracies1 = test(args, model, device, train_graphs, test_graphs, epoch,K)
           # output_epoch = str(epoch)
           # output_avg_loss = str(avg_loss)
           # output_acc_train = str(acc_train)
           # output_acc_test = str(acc_test)
           # output_accuracies1 = str(accuracies1)
           # output_string_avg_loss = output_avg_loss+'\n'
           # output_string_acc_train = output_acc_train + '\n'
           # output_string_acc_test = output_acc_test + '\n'
           # output_string_accuracies1 = output_accuracies1 + '\n'
           # path1 = "output_avg_loss.txt"
           # path2 = "output_acc_train.txt"
           # path3 = "output_acc_test.txt"
           # path4 = "output_accuracies1.txt"
           # file = open(path1,"a")
           # file.write(output_string_avg_loss)
           # file = open(path2, "a")
           # file.write(output_string_acc_train)
           # file = open(path3, "a")
           # file.write(output_string_acc_test)
           # file = open(path4, "a")
           # file.write(output_string_accuracies1)

           if (accuracies1 > maxacc1):
               maxacc1 = accuracies1
               if (maxacc1 == 1.0):
                   break

           if not args.filename == "":
               with open(args.filename, 'w') as f:
                   f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                   f.write("\n")
           print("")
           print("maxacc1maxacc1maxacc1", maxacc1)
           print(model.eps)
       # print("maxaccmaxaccmaxacc", maxacc)
       # print("maxepoch", maxepoch)
       print("maxacc1maxacc1maxacc1", maxacc1)
       end_fold = time.time()
       runtime_fold = end_fold - start
       print("运行时间_fold：", runtime_fold)
       # print("maxepoch1", maxepoch1)
       # print("maxaccuracies", maxaccuracies)
       maxaccuracies1.append(maxacc1)
       print("maxaccuracies1", maxaccuracies1)
   end = time.time()
   runtime = end - start
   print("运行时间：", runtime)
if __name__ == '__main__':
   main()
# LAGNN+点积+WL用于非DD数据集