import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models
import math
import numpy as np
import utils
from torch.distributions.bernoulli import Bernoulli
import random


class rnn_encoder(nn.Module):

    def __init__(self, config, embedding=None, padding_idx=0):
        super(rnn_encoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(
            config.src_vocab_size, config.emb_size, padding_idx=padding_idx)
        self.hidden_size = config.hidden_size
        self.config = config

        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)
            self.dropout = nn.Dropout(config.dropout)
            self.emb_drop = nn.Dropout(config.emb_dropout)

    def forward(self, inputs, lengths):

        embs = pack(self.emb_drop(self.embedding(inputs)), lengths) # [max_src_len, batch_size, emb_dim]
        # outputs: [max_src_len, batch_size, hidden_size * num_directions]
        # state = (h_n, c_n)
        # h_n: [num_layers * num_directions, batch_size, hidden_size]
        # c_n: [num_layers * num_directions, batch_size, hidden_size]
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if self.config.bidirectional:
            # outputs: [max_src_len, batch_size, hidden_size]
            outputs = outputs[:, :, :self.config.hidden_size] + \
                outputs[:, :, self.config.hidden_size:] 
            outputs = self.dropout(outputs)

        if self.config.cell == 'gru':
            state = state[:self.config.dec_num_layers] 
        else:
            state = (state[0][::2], state[1][::2]) 

        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True, padding_idx=0):
        super(rnn_decoder, self).__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(
            config.tgt_vocab_size, config.emb_size, padding_idx=padding_idx)

        input_size = config.emb_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(2 * config.hidden_size, config.tgt_vocab_size)
        self.linear_ = nn.Linear(config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        if not use_attention or config.attention == 'None':
            self.attention1 = None
            self.attention2 = None
        elif config.attention == 'bahdanau':
            self.attention1 = models.bahdanau_attention(
                config.hidden_size, config.emb_size, config.pool_size)
            self.attention2 = models.bahdanau_attention(
                config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention1 = models.luong_attention(
                config.hidden_size, config.emb_size, config.pool_size)
            self.attention2 = models.luong_attention(
                config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention1 = models.luong_gate_attention(
                config.hidden_size, config.emb_size, prob=config.dropout)
            self.attention2 = models.luong_gate_attention(
                config.hidden_size, config.emb_size, prob=config.dropout)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.emb_drop = nn.Dropout(config.emb_dropout)
        self.config = config

    def forward(self, input, state):
        embs = self.emb_drop(self.embedding(input)) # [batch_size, emb_dim]
        # output: [batch_size, hidden_size]
        # state: (h_t, c_t) for LSTM or h_t for GRU, [dec_num_layers, batch_size, hidden_size]
        output, state = self.rnn(embs, state)

        if self.attention1 is not None:
            if self.config.attention == 'luong_gate':
                output1, attn_weights1 = self.attention1(output)
            else:
                output1, attn_weights1 = self.attention1(output, embs)
        else:
            attn_weights1 = None

        if self.attention2 is not None:
            if self.config.attention == 'luong_gate':
                output2, attn_weights2 = self.attention2(output)
            else:
                output2, attn_weights2 = self.attention2(output, embs)
        else:
            attn_weights2 = None

        output = self.compute_score(torch.cat([output1, output2], dim=-1))

        return output, state, attn_weights1

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores


class StackedLSTM(nn.Module):

    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1
