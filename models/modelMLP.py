#     modelMLP.py
#     The class for the MLP model.
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

        # if conf.useP0:
        #     self.fcIn = nn.Linear(conf.dimP*2, conf.hiddenDim)
        # else:
        #     self.fcIn = nn.Linear(conf.dimP, conf.hiddenDim)
        if conf.has("allPt") and conf.allPt:
            inputDim = conf.dimP * (conf.numT+1)
        else:
            inputDim = conf.dimP

        self.fcIn = nn.Linear(inputDim, conf.hiddenDim)
        self.fcH = nn.ModuleList([])
        for _ in range(conf.hiddenLayers):
            self.fcH.append(nn.Linear(conf.hiddenDim, conf.hiddenDim))
        self.fcOut = nn.Linear(conf.hiddenDim, conf.dimPg)

        if conf.dropout > 0:
            self.dropout = nn.Dropout(p=conf.dropout)

    def _activation(self, x):
        if self.conf.activation == 'sigmoid':
            #x = F.sigmoid(x)
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
            'weight/fcIn': self.fcIn.weight.clone().data.cpu().numpy().flatten(),
            'bias/fcIn': self.fcIn.bias.clone().data.cpu().numpy(),
            'weight/fcOut': self.fcOut.weight.clone().data.cpu().numpy().flatten(),
            'bias/fcOut': self.fcOut.bias.clone().data.cpu().numpy(),
        }
        for i in range(len(self.fcH)):
            hists['weight/fcH{}'.format(i)] = self.fcH[i].weight.clone().data.cpu().numpy().flatten()
            hists['bias/fcH{}'.format(i)] = self.fcH[i].bias.clone().data.cpu().numpy()

        return hists

    def forward(self, true, returnHists=False):
        P = true['P']
        hists = {}

        # Psplit = P.split(1, dim=1)
        # if self.conf.useP0:
        #     x = self.fcIn(torch.cat([Psplit[0], Psplit[-1]], dim=2))
        # else:
        #     x = self.fcIn(Psplit[-1])

        if self.conf.has("allPt") and self.conf.allPt:
            P = P.reshape(P.shape[0], -1)
        else:
            P = P[:,-1,:]

        x = self.fcIn(P)

        if returnHists:
            hists['out/fcIn'] = x.clone().data.cpu().numpy().flatten()

        x = self._activation(x)

        if returnHists:
            hists['act/fcIn'] = x.clone().data.cpu().numpy().flatten()

        for i in range(len(self.fcH)):
            x = self.fcH[i](x)
            if returnHists:
                hists['out/fcH{}'.format(i)] = x.clone().data.cpu().numpy().flatten()
            x = self._activation(x)
            if returnHists:
                hists['act/fcH{}'.format(i)] = x.clone().data.cpu().numpy().flatten()
            if self.conf.dropout > 0:
                x = self.dropout(x)

        x = self.fcOut(x)
        if returnHists:
            hists['out/fcOut'] = x.clone().data.cpu().numpy().flatten()
        if self.conf.loss in ['kldAvg', 'kldFirstP', 'kldFirstPg']:
            x = torch.log_softmax(x, dim=2)

        if returnHists:
            return {'Pg':x}, hists
        else:
            return {'Pg':x}
