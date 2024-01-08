# -*- coding: utf-8 -*-
"""
v2.0 修复RNN参数初始化不当，引起的时间步传播梯度消失问题。   2023.04.12
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

#
class LSTM_Cell(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, token_dim, hidden_dim
                 , input_act=nn.ReLU()
                 , forget_act=nn.ReLU()
                 , output_act=nn.ReLU()
                 , hatcell_act=nn.Tanh()
                 , hidden_act=nn.Tanh()
                 , device="cpu"):
        super().__init__()
        #
        self.hidden_dim = hidden_dim
        self.device = device
        #
        self.InputG = Simple_RNN_Cell(
            token_dim, hidden_dim, activation=input_act, device=device
        )
        self.ForgetG = Simple_RNN_Cell(
            token_dim, hidden_dim, activation=forget_act, device=device
        )
        self.OutputG = Simple_RNN_Cell(
            token_dim, hidden_dim, activation=output_act, device=device
        )
        self.HatCell = Simple_RNN_Cell(
            token_dim, hidden_dim, activation=hatcell_act, device=device
        )
        self.HiddenActivation = hidden_act.to(self.device)

    def forward(self, inputs, last_state):
        """
        inputs:      it is the word vector of this time step token.
        last_state:  last_state = [last_cell, last_hidden_state]
        :return:
        """
        Ig = self.InputG(
            inputs, last_state
        )[-1]
        Fg = self.ForgetG(
            inputs, last_state
        )[-1]
        Og = self.OutputG(
            inputs, last_state
        )[-1]
        hat_cell = self.HatCell(
            inputs, last_state
        )[-1]
        cell = Fg * last_state[0] + Ig * hat_cell
        hidden = Og * self.HiddenActivation(cell)
        return [cell, hidden]

    def zero_initialization(self, batch_size):
        init_cell = torch.zeros([batch_size, self.hidden_dim]).to(self.device)
        init_state = torch.zeros([batch_size, self.hidden_dim]).to(self.device)
        return [init_cell, init_state]


#
class RNN_Layer(nn.Module):
    """
    bidirectional:  If ``True``, becomes a bidirectional RNN network. Default: ``False``.
    padding:        String, 'pre' or 'post' (optional, defaults to 'pre'): pad either before or after each sequence.
    """
    def __init__(self, rnn_cell, bidirectional=False, pad_position='post'):
        super().__init__()
        self.RNNCell = rnn_cell
        self.bidirectional = bidirectional
        self.padding = pad_position

    def forward(self, inputs, mask=None, initial_state=None):
        """
        inputs:   it's shape is [batch_size, time_steps, token_dim]
        mask:     it's shape is [batch_size, time_steps]
        :return
        sequence:    it is hidden state sequence, and its' shape is [batch_size, time_steps, hidden_dim]
        last_state: it is the hidden state of input sequences at last time step,
                    but, attentively, the last token wouble be a padding token,
                    so this last state is not the real last state of input sequences;
                    if you want to get the real last state of input sequences, please use utils.get_rnn_last_state(hidden state sequence).
        """
        batch_size, time_steps, token_dim = inputs.shape
        #
        if initial_state is None:
            initial_state = self.RNNCell.zero_initialization(batch_size)
        if mask is None:
            if batch_size == 1:
                mask = torch.ones([1, time_steps]).to(inputs.device.type)
            elif self.padding == 'pre':
                raise ValueError('请给定掩码矩阵(mask)')
            elif self.padding == 'post' and self.bidirectional is True:
                raise ValueError('请给定掩码矩阵(mask)')

        # 正向时间步循环
        hidden_list = []
        hidden_state = initial_state
        last_state = None
        for i in range(time_steps):
            hidden_state = self.RNNCell(inputs[:, i], hidden_state)
            hidden_list.append(hidden_state[-1])
            if i == time_steps - 1:
                """获取最后一时间步的输出隐藏状态"""
                last_state = hidden_state
            if self.padding == 'pre':
                """如果padding值填充在序列尾端，则正向时间步传播应加 mask 操作"""
                hidden_state = [
                    hidden_state[j] * mask[:, i:i + 1] + initial_state[j] * (1 - mask[:, i:i + 1])  # 重新初始化（加数项作用）
                    for j in range(len(hidden_state))
                ]
        sequence = torch.reshape(
            torch.unsqueeze(
                torch.concat(hidden_list, dim=1)
                , dim=1)
            , [batch_size, time_steps, -1]
        )

        # 反向时间步循环
        if self.bidirectional is True:
            hidden_list = []
            hidden_state = initial_state
            for i in range(time_steps, 0, -1):
                hidden_state = self.RNNCell(inputs[:, i - 1], hidden_state)
                hidden_list.insert(0, hidden_state[-1])
                if i == time_steps:
                    """获取最后一时间步的cell_state"""
                    last_state = [
                        torch.concat([last_state[j], hidden_state[j]], dim=1)
                        for j in range(len(hidden_state))
                    ]
                if self.padding == 'post':
                    """如果padding值填充在序列首端，则正反时间步传播应加 mask 操作"""
                    hidden_state = [
                        hidden_state[j] * mask[:, i - 1:i] + initial_state[j] * (1 - mask[:, i - 1:i])  # 重新初始化（加数项作用）
                        for j in range(len(hidden_state))
                    ]
            sequence = torch.concat([
                sequence,
                torch.reshape(
                    torch.unsqueeze(
                        torch.concat(hidden_list, dim=1)
                        , dim=1)
                    , [batch_size, time_steps, -1]
                )
            ], dim=-1)

        return sequence, last_state


