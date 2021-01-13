#     modelRNN.py
#     The class for the RNN model.
#     Copyright (C) 2021  Stefano Martina (stefano.martina@unifi.it)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, conf):
        super(Model, self).__init__()

        self.conf = conf

        self.aggregation = conf.aggregation if conf.has("aggregation") else "cat"

        bidirectional = conf.bidirectional if conf.has("bidirectional") else True
        self.bidirectional = bidirectional

        rnnType = conf.rnnType if conf.has("rnnType") else "GRU"
        if rnnType == "GRU":
            RNN = nn.GRU
        elif rnnType == "LSTM":
            RNN = nn.LSTM
        else:
            raise ValueError("RNN type {} not valid".format(rnnType))

        self.rnn = RNN(input_size=conf.dimP, hidden_size=conf.hiddenDim, num_layers=conf.hiddenLayers, batch_first=True, dropout=(conf.dropout if conf.hiddenLayers>1 else 0), bidirectional=bidirectional)
        dimOut = conf.hiddenDim * 2 if bidirectional else conf.hiddenDim

        if self.aggregation == "attention":
            self.attentionLayer = nn.Linear(dimOut, conf.attentionDim)
            self.contextVector = nn.Parameter(torch.randn(conf.attentionDim))
            self.contextVector.requires_grad=True

        if conf.dropout > 0:
            self.dropout = nn.Dropout(p=conf.dropout)

        self.fcOut = nn.Linear(dimOut, conf.dimPg)

    def _activation(self, x):
        if self.conf.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.conf.activation == 'tanh':
            x = torch.tanh(x)
        elif self.conf.activation == 'relu':
            x = F.relu(x)
        elif not self.conf.activation == 'none':
            raise Exception("value of activationH not valid")

        return x

    def getWeightsHists(self):
        hists = {
            'weight/fcOut': self.fcOut.weight.clone().data.cpu().numpy().flatten(),
            'bias/fcOut': self.fcOut.bias.clone().data.cpu().numpy(),
        }
        for i in range(self.conf.hiddenLayers):
            W_ih, W_hh, b_ih, b_hh = (self.rnn.all_weights[i][j].clone().data.cpu().numpy() for j in range(4))

            hists['weight/rnn{}-W_ir'.format(i)] = W_ih[:self.conf.hiddenDim,:].flatten()
            hists['weight/rnn{}-W_iz'.format(i)] = W_ih[self.conf.hiddenDim:self.conf.hiddenDim*2,:].flatten()
            hists['weight/rnn{}-W_in'.format(i)] = W_ih[self.conf.hiddenDim*2:,:].flatten()

            hists['weight/rnn{}-W_hr'.format(i)] = W_hh[:self.conf.hiddenDim,:].flatten()
            hists['weight/rnn{}-W_hz'.format(i)] = W_hh[self.conf.hiddenDim:self.conf.hiddenDim*2,:].flatten()
            hists['weight/rnn{}-W_hn'.format(i)] = W_hh[self.conf.hiddenDim*2:,:].flatten()

            hists['bias/rnn{}-b_ir'.format(i)] = b_ih[:self.conf.hiddenDim]
            hists['bias/rnn{}-b_iz'.format(i)] = b_ih[self.conf.hiddenDim:self.conf.hiddenDim*2]
            hists['bias/rnn{}-b_in'.format(i)] = b_ih[self.conf.hiddenDim*2:]

            hists['bias/rnn{}-b_hr'.format(i)] = b_hh[:self.conf.hiddenDim]
            hists['bias/rnn{}-b_hz'.format(i)] = b_hh[self.conf.hiddenDim:self.conf.hiddenDim*2]
            hists['bias/rnn{}-b_hn'.format(i)] = b_hh[self.conf.hiddenDim*2:]

        if self.aggregation == "attention":
            hists['weight/fcAtt'] = self.attentionLayer.weight.clone().data.cpu().numpy().flatten()
            hists['bias/fcAtt'] = self.attentionLayer.bias.clone().data.cpu().numpy()
            hists['weight/context'] = self.contextVector.clone().data.cpu().numpy()

        return hists

    def forward(self, true, returnHists=False):
        P = true['P']
        hists = {}

        x,_ = self.rnn(P)

        if self.aggregation == "max":
            x = torch.max(x, dim=1)[0]  #temporal max
        elif self.aggregation == "cat":
            if self.bidirectional: # take last for forward and first for backward
                s = x.shape
                x = x.view(s[0],s[1],2,-1)
                x = torch.cat((x[:,-1,0,:], x[:,0,1,:]), axis=1)
            else: #take only last
                x = x[:,-1,:]
        elif self.aggregation == "attention":
            a = self.attentionLayer(x)
            a = torch.tanh(a)
            a = torch.matmul(a, self.contextVector)
            a = torch.softmax(a,1)
            #if getAttention and self.conf.getAttentionType == "weights":
            #    statusPhrase = a.clone().detach()
            a = a[:,:,None]*x
            #if getAttention and self.conf.getAttentionType == "weighted":
            #    statusPhrase = a.clone().detach()
            x = torch.sum(a, dim=1)
        else:
            raise ValueError("Aggregation {} not valid".format(self.aggregation))

        if returnHists:
            hists['out/rnn'] = x.clone().data.cpu().numpy().flatten()

        x = self._activation(x)

        if returnHists:
            hists['act/rnn'] = x.clone().data.cpu().numpy().flatten()

        if self.conf.dropout > 0:
            x = self.dropout(x)

        x = self.fcOut(x)

        if returnHists:
            hists['out/fcOut'] = x.clone().data.cpu().numpy().flatten()

        if self.conf.loss in ['kldAvg', 'kldFirstP', 'kldFirstPg']:
            x = torch.log_softmax(x, dim=1)

        x = x.view([x.shape[0], 1, x.shape[1]])

        if returnHists:
            return {'Pg':x}, hists
        else:
            return {'Pg': x}
