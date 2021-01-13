#     loaders.py
#     The class and function to create the Pytorch dataset and dataloader.
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
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os

class DatasetV1(torch.utils.data.Dataset):
    def __init__(self, conf, mySet=None):
        self.conf = conf

        if conf.has("useTune") and conf.useTune:
            basePath = "../../../data"
        else:
            basePath = "data"

        fileDataset = np.load(os.path.join(basePath,conf.dataset))
        fileSplit = np.load(os.path.join(basePath,conf.split))

        if conf.has('yKey'):
            yKey = conf.yKey
        else:
            if conf.loss in ['bce', 'cce']:
                yKey = 'Pg0c'
            else:
                yKey = 'Pg'


        self.data = {}

        P = fileDataset['P']
        Pg = fileDataset[yKey]    #use always Pg as dataset key

        if not mySet is None:
            split = fileSplit[mySet]
        else:
            split = [i for s in [fileSplit[k] for k in fileSplit] for i in s]

        Ps = P[split]
        if conf.normalizeP is True: #TODO: maybe normalize only with mean/std of train
            self.data['P'] = (Ps - Ps.mean(axis=0)) / Ps.std(axis=0)
        elif conf.normalizeP == 'minMax':
            self.data['P'] = (Ps - np.min(Ps)) / (np.max(Ps) - np.min(Ps))
        elif conf.normalizeP is False:
            self.data['P'] = Ps
        else:
            raise ValueError("normalizeP not valid")

        self.data['Pg'] = Pg[split]


        if 'TM' in fileDataset.files:
            TM = fileDataset['TM']
            self.data['TM'] = TM[split]
        if 'TM2' in fileDataset.files:
            TM2 = fileDataset['TM2']
            self.data['TM2'] = TM2[split]
            
        if 'rho' in fileDataset.files:#TODO: implement rho normalization
            rho = fileDataset['rho']
            rhor = np.stack((rho.real, rho.imag), axis = 2)
            self.data['rho'] = rhor[split]


    def __len__(self):
        return len(self.data['P'])

    def __getitem__(self, idx):
        #P = torch.from_numpy(self.P[idx])
        P = torch.Tensor(self.data['P'][idx])
        if self.conf.loss == 'bce':
            if not self.conf.alreadyBinarized:
                Pg = torch.Tensor(np.zeros((self.conf.dimPg)))
                if self.conf.dimPg == 1: #if binary
                    Pg[0] = self.data['Pg'][idx]
                else:
                    Pg[self.data['Pg'][idx]] = 1.
            else:
                Pg = torch.Tensor(self.data['Pg'][idx])
        elif self.conf.loss == 'cce':
            Pg = torch.LongTensor([self.data['Pg'][idx]])
        else:
            Pg = torch.Tensor(self.data['Pg'][idx])

        toReturn = {
            'P':    P,
            'Pg':   Pg,
        }

        if 'TM' in self.data:
            TM = torch.Tensor(self.data['TM'][idx])
            toReturn['TM'] = TM
        if 'TM2' in self.data:
            TM2 = torch.Tensor(self.data['TM2'][idx])
            toReturn['TM2'] = TM2
            
        if 'rho' in self.data:
            rho = torch.Tensor(self.data['rho'][idx])
            toReturn['rho'] = rho

        return toReturn

def v1(conf, all=False):
    if all:
        datasets = DatasetV1(conf)
        loaders = torch.utils.data.DataLoader(datasets, batch_size=conf.batchSize, shuffle=True)
    else:
        splits = ['train', 'valid', 'test']
    
        datasets = {s: DatasetV1(conf, s) for s in splits}
        loaders = {s: torch.utils.data.DataLoader(datasets[s], batch_size=conf.batchSize, shuffle=True) for s in splits}
    
    return loaders, datasets
