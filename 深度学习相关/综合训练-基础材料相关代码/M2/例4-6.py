# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:29:41 2022

@author: bruce dee
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


weight_ih = torch.tensor([[ 0.3162,  0.0833,  0.1223,  0.4317, -0.2017,  0.1417, -0.1990,  0.3196,
          0.3572, -0.4123],
        [ 0.3818,  0.2136,  0.1949,  0.1841,  0.3718, -0.0590, -0.3782, -0.1283,
         -0.3150,  0.0296],
        [-0.0835, -0.2399, -0.0407,  0.4237, -0.0353,  0.0142, -0.0697,  0.0703,
          0.3985,  0.2735],
        [ 0.1587,  0.0972,  0.1054,  0.1728, -0.0578, -0.4156, -0.2766,  0.3817,
          0.0267, -0.3623],
        [ 0.0705,  0.3695, -0.4226, -0.3011, -0.1781,  0.0180, -0.1043, -0.0491,
         -0.4360,  0.2094],
        [ 0.3925,  0.2734, -0.3167, -0.3605,  0.1857,  0.0100,  0.1833, -0.4370,
         -0.0267,  0.3154],
        [ 0.2075,  0.0163,  0.0879, -0.0423, -0.2459, -0.1690, -0.2723,  0.3715,
          0.2461,  0.1564],
        [-0.3429,  0.3451,  0.1402,  0.3094, -0.1759,  0.0948,  0.4367,  0.3008,
          0.3587, -0.0939],
        [ 0.3407, -0.3503,  0.0387, -0.2518, -0.1043, -0.1145,  0.0335,  0.4070,
          0.2214, -0.0019],
        [ 0.3175, -0.2292,  0.2305, -0.0415, -0.0778,  0.0524, -0.3426,  0.0517,
          0.1504,  0.3823],
        [-0.1392,  0.1610,  0.4470, -0.1918,  0.4251, -0.2220,  0.1971,  0.1752,
          0.1249,  0.3537],
        [-0.1807,  0.1175,  0.0025, -0.3364, -0.1086, -0.2987,  0.1977,  0.0402,
          0.0438, -0.1357],
        [ 0.0022, -0.1391,  0.1285,  0.4343,  0.0677, -0.1981, -0.2732,  0.0342,
         -0.3318, -0.3361],
        [-0.2911, -0.1519,  0.0331,  0.3080,  0.1732,  0.3426, -0.2808,  0.0377,
         -0.3975,  0.2565],
        [ 0.0932,  0.4326, -0.3181,  0.3586,  0.3775,  0.3616,  0.0638,  0.4066,
          0.2987,  0.3337]])
weight_hh = torch.tensor([[-0.0291, -0.3432, -0.0056,  0.0839, -0.3046],
        [-0.2565, -0.4288, -0.1568,  0.3896,  0.0765],
        [-0.0273,  0.0180,  0.2789, -0.3949, -0.3451],
        [-0.1487, -0.2574,  0.2307,  0.3160, -0.4339],
        [-0.3795, -0.4355,  0.1687,  0.3599, -0.3467],
        [-0.2070,  0.1423, -0.2920,  0.3799,  0.1043],
        [-0.1245,  0.0290,  0.1394, -0.1581, -0.3465],
        [ 0.0030,  0.0081,  0.0090, -0.0653,  0.2871],
        [-0.1248, -0.0433,  0.1839, -0.2815,  0.1197],
        [-0.0989,  0.2145, -0.2426,  0.0165,  0.0438],
        [-0.3598, -0.3252,  0.1715, -0.1302,  0.2656],
        [-0.4418, -0.2211, -0.3684,  0.1786, -0.0130],
        [-0.0834, -0.0744, -0.3496,  0.1268,  0.0111],
        [-0.3086,  0.1683, -0.0090, -0.4325,  0.2406],
        [ 0.2392, -0.0843, -0.3088,  0.0180,  0.3375]])
bias_ih = torch.tensor([ 0.4094, -0.3376, -0.2020,  0.3482,  0.2186,  0.2768, -0.2226,  0.3853,
        -0.3676, -0.0215,  0.0093,  0.0751, -0.3375,  0.4103,  0.4395])
bias_hh = torch.tensor([-0.3088,  0.0165, -0.2382,  0.4288,  0.2494,  0.2634,  0.1443, -0.0445,
         0.2518,  0.0076, -0.1631,  0.2309,  0.1403, -0.1159, -0.1226])

class GRUtest(nn.Module):     # pytorch中的gru
    def __init__(self, input, hidden, act):
        super().__init__()
        self.gru = nn.GRU(input, hidden, batch_first=True)
        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):  
        self.gru.flatten_parameters()
        gru_out, gru_state = self.gru(x)   
        return gru_out, gru_state

class GRULayer:
    def __init__(self, input_size, hidden_size, act):
        self.bias_ih = bias_ih.reshape(-1)
        self.bias_hh = bias_hh.reshape(-1)
        self.weight_ih = weight_ih.reshape(-1)
        self.weight_hh = weight_hh.reshape(-1)
        self.nb_input = input_size
        self.nb_neurons = hidden_size
        self.activation = act


def compute_gru(gru, state, input):
    M = gru.nb_input
    N = gru.nb_neurons
    r = torch.zeros(N)
    z = torch.zeros(N)
    n = torch.zeros(N)
    h_new = torch.zeros(N)
    
    for i in range(N):
        sum = gru.bias_ih[0*N + i] +  gru.bias_hh[0*N + i]
        for j in range(M):
            sum += input[j] * gru.weight_ih[0*M*N + i*M + j]
        for j in range(N):
            sum += state[j] * gru.weight_hh[0*N*N + i*N + j] 
        r[i] = torch.sigmoid(sum)
    
    for i in range(N):
        sum = gru.bias_ih[1*N+i] +  gru.bias_hh[1*N+i]
        for j in range(M):
            sum += input[j] * gru.weight_ih[1*M*N + i*M + j]
        for j in range(N):
            sum += state[j] * gru.weight_hh[1*N*N + i*N + j] 
        z[i] = torch.sigmoid(sum)
    
    for i in range(N):
        sum = 0
        sum += gru.bias_ih[2*N+i]
        tmp = 0 
        for j in range(M):
            sum += input[j] * gru.weight_ih[2*M*N + i*M + j]
        for j in range(N):
            tmp += state[j] * gru.weight_hh[2*N*N + i*N + j]
        sum += r[i]*(tmp + gru.bias_hh[2*N+i])
        n[i] = torch.tanh(sum)
    
    for i in range(N):
        h_new[i] = (1 - z[i]) * n[i] + z[i] * state[i]
        state[i] = h_new[i]

b = torch.randn((1, 5, 10))
   

if __name__ == '__main__':
    insize = 10
    hsize = 5
    net1 = GRUtest(insize, hsize, 'tanh')
    model_ckpt1 = torch.load('./nn_test.pkl')    #根据路径需要及进行修改
    net1.load_state_dict(model_ckpt1.state_dict())
    gru = GRULayer(insize, hsize, 'tanh')      # 自己写的gru类，包含gru参数 
    out = torch.zeros((5, 5))    #用以保存计算结果
    state = torch.zeros(5)       #用来储存gidden_state的变量，初始化为0
    for i in range(5):
        input = b[0][i]
        compute_gru(gru, state, input)
        out[i] = state
    print("自己实现前向计算结果：")
    print(out)
    print("pytorch实现前向计算结果：")
    torch_out, _ = net1(b)
    print(torch_out)

