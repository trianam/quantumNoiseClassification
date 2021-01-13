#!/usr/bin/env python
# coding: utf-8

#     svm.py
#     To train SVM (uncomment correct line in main).
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


from sklearn import svm
from sklearn import preprocessing
import numpy as np
import os


def extractData(dataset, split, normalize=False, takeN=-1):
    print("Dataset: {}".format(dataset))
    print("Split: {}".format(split))
    print("Normalization: {}".format(normalize))
    print("TakeN: {}".format(takeN))

    fileDataset = np.load(os.path.join('data', dataset))
    fileSplit = np.load(os.path.join('data', split))

    data = {k:{} for k in fileSplit.keys()}

    P = fileDataset['P']
    Pg = fileDataset['Pg0c']

    for k in fileSplit.keys():
        split = fileSplit[k]
        #split = [i for s in [fileSplit[k] for k in fileSplit] for i in s]

        #debug
        # split = split[:1000]

        data[k]['P'] = P[split]
        if takeN is None:
            data[k]['P'] = data[k]['P'].reshape(data[k]['P'].shape[0],-1) #take all
        else:
            data[k]['P'] = data[k]['P'][:,takeN,:]
        data[k]['Pg'] = Pg[split]

    if normalize == 'meanStd':
        trainMean = data['train']['P'].mean(axis=0)
        trainStd = data['train']['P'].std(axis=0)
        for k in data:
            data[k]['P'] = (data[k]['P'] - trainMean) / trainStd

    elif normalize == 'minMax':
        trainMin = np.min(data['train']['P'])
        trainMax = np.max(data['train']['P'])
        for k in data:
            data[k]['P'] = (data[k]['P'] - trainMin) / (trainMax - trainMin)

    elif normalize == 'standardScaler':
        scaler = preprocessing.StandardScaler().fit(data['train']['P'])
        for k in data:
            data[k]['P'] = scaler.transform(data[k]['P'])

    elif normalize is not False:
        raise ValueError("normalizeP not valid")

    return data

def train(clf, data):
    clf.fit(data['train']['P'], data['train']['Pg'])

    pred = {k:clf.predict(data[k]['P']) for k in data.keys()}
    acc = {k: np.sum(pred[k] == data[k]['Pg']) / len(data[k]['Pg']) for k in data.keys()}
    print("Train acc.: {}; valid acc.: {}; test acc.: {}".format(acc['train'], acc['valid'], acc['test']))


if __name__ == "__main__":
    # normalize = "meanStd"
    # normalize = "minMax"
    # normalize = "standardScaler"
    normalize = False

    # data = extractData("datasetIIDV2-Pg0cBinarySeparated.npz", "splitIIDV2-Pg0cBinarySeparated.npz", normalize)
    # data = extractData("datasetIIDV2-Pg0cBinarySeparated-1rep.npz", "splitIIDV2-Pg0cBinarySeparated-1rep.npz", normalize)
    # data = extractData("datasetIIDV2-Pg0cBinarySeparated-1rep2.npz", "splitIIDV2-Pg0cBinarySeparated-1rep.npz", normalize)
    # data = extractData("datasetIIDV2-Pg0cBinarySeparated-1rep-float64.npz", "splitIIDV2-Pg0cBinarySeparated-1rep.npz", normalize)
    # data = extractData("datasetIIDV2-Pg0cBinarySeparated5.npz", "splitIIDV2-Pg0cBinarySeparated5.npz", normalize)
    # data = extractData("datasetIID-scratch-t0.015.npz", "splitIID-scratch-t0.015.npz", normalize)
    # data = extractData("datasetIID-scratch-t0.015-small.npz", "splitIID-scratch-t0.015-small.npz", normalize)
    # data = extractData("datasetIIDb-scratch-t0.015-small.npz", "splitIIDb-scratch-t0.015-small.npz", normalize)
    # data = extractData("datasetIIDb-scratch-t0.015-small.npz", "splitIIDb-scratch-t0.015-small.npz", normalize, takeN=None)
    # data = extractData("datasetIIDb-scratch-t0.015.npz", "splitIIDb-scratch-t0.015.npz", normalize)
    # data = extractData("datasetIIDvsM1-scratch-t0.015.npz", "splitIIDvsM1-scratch-t0.015.npz", normalize)
    # data = extractData("datasetIIDb-scratch-t0.1/rep01.npz", "splitIIDb-scratch-t0.1.npz", normalize)
    # data = extractData("datasetIIDb-scratch-t0.1/rep01.npz", "splitIIDb-scratch-t0.1.npz", normalize, takeN=None)
    # data = extractData("datasetIIDb-scratch-t1/rep01.npz", "splitIIDb-scratch-t1.npz", normalize)
    # data = extractData("datasetIIDb-scratch-t0.01/rep01.npz", "splitIIDb-scratch-t0.01.npz", normalize)
    # data = extractData("datasetIIDb-scratch-t0.01/rep01.npz", "splitIIDb-scratch-t0.01.npz", normalize, takeN=None)
    # data = extractData("datasetIIDb-scratch-t1/rep01.npz", "splitIIDb-scratch-t1.npz", normalize, takeN=None)
    # data = extractData("datasetM1-scratch-t1/rep01.npz", "datasetM1-scratch-t1/split-rep01.npz", normalize)
    # data = extractData("datasetM1-scratch-t1/rep01.npz", "datasetM1-scratch-t1/split-rep01.npz", normalize, takeN=None)
    # data = extractData("datasetIIDvsM1-scratch-t1/rep01.npz", "datasetIIDvsM1-scratch-t1/split-rep01.npz", normalize)
    # data = extractData("datasetIIDvsM1-scratch-t1/rep01.npz", "datasetIIDvsM1-scratch-t1/split-rep01.npz", normalize, takeN=None)

    # data = extractData("datasetM1-scratch-t0.1/rep01.npz", "datasetM1-scratch-t0.1/split-rep01.npz", normalize)
    # data = extractData("datasetM1-scratch-t0.1/rep01.npz", "datasetM1-scratch-t0.1/split-rep01.npz", normalize, takeN=None)
    # data = extractData("datasetIIDvsM1-scratch-t0.1/rep01.npz", "datasetIIDvsM1-scratch-t0.1/split-rep01.npz", normalize)
    data = extractData("datasetIIDvsM1-scratch-t0.1/rep01.npz", "datasetIIDvsM1-scratch-t0.1/split-rep01.npz", normalize, takeN=None)

    print("Linear SVM (LinearSVC)")
    train(svm.LinearSVC(), data)

    print("Linear SVM")
    train(svm.SVC(kernel='linear'), data)

    print("Poly d.2 SVM")
    train(svm.SVC(kernel='poly', degree=2), data)

    print("Poly d.3 SVM")
    train(svm.SVC(kernel='poly', degree=3), data)

    print("Poly d.4 SVM")
    train(svm.SVC(kernel='poly', degree=4), data)

    print("RBF SVM")
    train(svm.SVC(kernel='rbf'), data)

