import torch
import torch.nn as nn
import torch.nn.functional as F

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
            num_layers：神经网络中的层数（不包括输入层）。如果num_layers=1，这将简化为线性模型。
            input_dim：输入特征的维度
            hidden_dim：所有图层上隐藏单位的维度
            output_dim：用于预测的类数
            设备：要使用的设备
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers
        # 如果小于一层则报错，等于一层是只有输入输出没有隐藏层，大于一就是一个输入层一个输出层，其他都是隐藏层
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            # 减2是因为有输入输出层，因为num_layers为2所以中间for循环不执行，
            # 输入输出层中间共用了一个hidden_dim,所以是一个输入一个隐藏一个输出
            # 输入隐藏构成一个连接层
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            # 隐藏输出构成一个连接层
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            # 每层进行归一化
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)