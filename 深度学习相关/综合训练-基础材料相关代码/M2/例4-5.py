# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 20:33:59 2022

@author: bruce dee
"""
"""
LSTM与GRU区别：
把BasicLSTMCell改成GRUCell，就变成了GRU网络
lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
"""
import torch
from torch import nn
import torch.nn.functional as F

class GRUtest(nn.Module):
    def __init__(self, input, hidden, act):
        super().__init__()
        self.gru = nn.GRU(input, hidden, batch_first=True)
        if act == 'sigmoid':             # 激活函数后面未使用
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):  
        self.gru.flatten_parameters()
        gru_out, gru_state = self.gru(x)   
        return gru_out, gru_state
    
if __name__ == '__main__':
    insize = 10
    hsize = 5
    net1 = GRUtest(insize, hsize, 'tanh')
    for name, parameters in net1.named_parameters():
        print(name)
        print(parameters)

