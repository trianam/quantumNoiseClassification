#     funPytorch.py
#     Collect the function used for the creation, training and evaluation of the models.
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


import os
import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sl
import sklearn.metrics
import pickle
import time
import math
import warnings
from datetime import datetime

import torch
import torch.nn as nn

import torchsummary
from tensorboardX import SummaryWriter

from collections import defaultdict

import loaders
import importlib

from ray import tune

def filesPath(conf):
    if conf.has("useTune") and conf.useTune:
        return "files"
    else:
        return os.path.join("files", conf.path)

def makeModel(conf, device):
    modelPackage = importlib.import_module("models."+conf.model)
    model = modelPackage.Model(conf)
    model = model.to(device)

    if conf.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=conf.learningRate, weight_decay=conf.weightDecay)
    elif conf.optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=conf.learningRate, weight_decay=conf.weightDecay)

    return model,optim

def summary(conf, model, dataloaders=None): #TODO: check this, P shape maybe not correct
    if not dataloaders is None:
        batch = next(iter(dataloaders['test']))
        shape = batch['P'][0].shape
    else:
        shape = (conf.numT+1, conf.dimP)

    torchsummary.summary(model, shape)

def processData(conf, all=False):
    return loaders.v1(conf, all)

def loadCorpus(conf):
    _, extension = os.path.splitext(conf.fileCorpus)
    if extension == '.p':
        return pickle.load(open(conf.fileCorpus, 'rb'))
    elif extension == '.npz':
        return np.load(conf.fileCorpus)

def myLoss(pred, true, conf, eval=False):
    # lossFun = nn.NLLLoss()
    if conf.loss == 'bce':
        lossFun = nn.BCEWithLogitsLoss()
    elif conf.loss == 'cce':
        lossFun = nn.CrossEntropyLoss()
    elif conf.loss == 'mse' or conf.loss == 'mse2':
        lossFun = nn.MSELoss()
    elif conf.loss == 'mae':
        lossFun = nn.L1Loss()
    elif conf.loss in ['kldAvg', 'kldFirstP', 'kldFirstPg']:
        lossFun = nn.KLDivLoss(reduction='batchmean')

    if conf.loss == 'bce':
        loss = lossFun(pred['Pg'][:, 0, :] if len(pred['Pg'].shape)==3 else pred['Pg'], true['Pg'])  # in this case true['Pg'] are one hot categories of Pg0
        if not eval:
            loss.backward()
    elif conf.loss == 'cce':
        loss = lossFun(pred['Pg'][:,0,:] if len(pred['Pg'].shape)==3 else pred['Pg'], true['Pg'].reshape(true['Pg'].shape[0])) #in this case true['Pg'] are categories of Pg0
        if not eval:
            loss.backward()
    elif conf.loss == 'mse2':
        loss = lossFun(pred['Pg'][:, 0, :] if len(pred['Pg'].shape)==3 else pred['Pg'], true['Pg'])  # in this case true['Pg'] are one hot categories of Pg0
        if not eval:
            loss.backward()
    else:
        if eval or conf.loss == 'kldAvg' or conf.loss == 'mse' or conf.loss == 'mae':
            losses = []
            for key in pred:
                # TODO: pay attention, this works only because P7 last (ignored) and Pg0 first (use only Pg0 on some models)
                for i in range(pred[key].shape[1]):
                    losses.append(lossFun(pred[key][:,i,:], true[key][:,i,:]))
            loss = sum(losses)/len(losses)
            if not eval:
                loss.backward()

        elif conf.loss == 'kldFirstP' or conf.loss == 'kldFirstPg':
            if conf.loss == 'kldFirstP':
                keys = ['P', 'TM', 'TM2', 'Pg']
            elif conf.loss == 'kldFirstPg':
                keys = ['Pg', 'TM', 'TM2', 'P']

            for key in keys:
                if key in pred:
                    for i in range(pred[key].shape[1]-1):
                        loss = lossFun(pred[key][:,i,:], true[key][:,i,:])
                        if not eval:
                            loss.backward(retain_graph=True)
                    loss = lossFun(pred[key][:, -1, :], true[key][:, -1, :])
                    if not eval:
                        if not key == keys[-1]: #not retain for last
                            loss.backward(retain_graph=True)
                        else:
                            loss.backward()
    if eval:
        return loss


def evaluate(conf, model, dataloader, savePredictions=None, calculateHists=False):
    device = next(model.parameters()).device
    hists = {}
    model.eval()
    with torch.no_grad():
        runningLoss = 0.
        runningMetrics = defaultdict(float)
        predictions = defaultdict(list)

        for batchIndex, data in enumerate(dataloader):
            true = {k: data[k].to(device) for k in data}

            if calculateHists and (not conf.has('histOnlyBatches') or batchIndex < conf.histOnlyBatches): #only on first histOnlyBatches
                pred, currHists = model(true, returnHists=True)
                for h in currHists:
                    if not h in hists:
                        hists[h] = currHists[h]
                    else:
                        hists[h] = np.concatenate((hists[h], currHists[h]))
            else:
                pred = model(true)

            loss = myLoss(pred, true, conf, eval=True)

            runningLoss += loss.item()

            if conf.loss == 'bce':
                myPred = torch.sigmoid(pred['Pg'][:, 0, :] if len(pred['Pg'].shape)==3 else pred['Pg'])
                if conf.trackMetric == 'rmse':
                    runningMetrics["rmse"] += sl.metrics.mean_squared_error(true['Pg'].cpu(), pred['Pg'][:, 0, :].cpu() if len(pred['Pg'].shape)==3 else pred['Pg'].cpu())
                else:
                    for i in range(true['Pg'].shape[1]):
                        #for j in range(true['Pg'].shape[0]):
                        #    runningMetrics["acc-{}".format(i)] += true['Pg'][j,i]>0.5 and myPred[j,i]>0.5
                        runningMetrics["acc-{}".format(i)] += torch.where((true['Pg'][:,i]>0.5) == (myPred[:,i]>0.5))[0].shape[0] / true['Pg'].shape[0]

                if not savePredictions is None:
                    predictions['Pg0'].append(true['Pg'].cpu().detach().numpy())
                    predictions['Pg0p'].append(myPred.cpu().detach().numpy())
            elif conf.loss == 'cce':
                myPred = torch.softmax(pred['Pg'][:,0,:] if len(pred['Pg'].shape)==3 else pred['Pg'], dim=1)
                runningMetrics["acc"] += torch.sum(true['Pg'].reshape(true['Pg'].shape[0]) == myPred.max(dim=1).indices).float() / true['Pg'].shape[0]
                #for i in range(myPred.shape[1]): #accuracies on singles
                #    runningMetrics["accS-{}".format(i)] += sum(true['Pg'].reshape(true['Pg'].shape[0]) == myPred.max(dim=1).indices)

                if not savePredictions is None:
                    predictions['Pg0'].append(true['Pg'].reshape(true['Pg'].shape[0]).cpu().detach().numpy())
                    predictions['Pg0p'].append(myPred.cpu().detach().numpy())
            elif conf.loss == 'mse2':
                if conf.trackMetric == 'rmse':
                    runningMetrics["rmse"] += sl.metrics.mean_squared_error(true['Pg'].cpu(), pred['Pg'][:, 0, :].cpu() if len(pred['Pg'].shape)==3 else pred['Pg'].cpu())
                if not savePredictions is None:
                    predictions['Pg0'].append(true['Pg'].cpu().detach().numpy())
                    predictions['Pg0p'].append((pred['Pg'][:, 0, :].cpu() if len(pred['Pg'].shape)==3 else pred['Pg'].cpu()).detach().numpy())
            else:
                for key in pred:
                    keyP = key + "p"
                    #TODO: pay attention, this works only because P7 last (ignored) and Pg0 first (use only Pg0 on some models)
                    for i in range(pred[key].shape[1]):
                        myPred = pred[key][:,i,:]
                        if conf.loss in ['kldAvg', 'kldFirstP', 'kldFirstPg']:
                            myPred = torch.exp(myPred)
                        if conf.trackMetric == 'kld':
                            runningMetrics["{}{}".format(key,i)] += sp.stats.entropy(true[key][:,i,:].cpu(), myPred.cpu(), axis=1).mean(axis=0)
                        elif conf.trackMetric == 'mse':
                            runningMetrics["{}{}".format(key,i)] += sl.metrics.mean_squared_error(true[key][:,i,:].cpu(), myPred.cpu())

                    if not savePredictions is None:
                        predictions[key].append(true[key].cpu().detach().numpy())
                        myPred = pred[key]
                        if conf.loss in ['kldAvg', 'kldFirstP', 'kldFirstPg']:
                            myPred = torch.exp(myPred) #exp to return distribution (model uses log_softmax)
                        predictions[keyP].append(myPred.cpu().detach().numpy())

        for k in runningMetrics:
            runningMetrics[k] /= len(dataloader)

        toReturn = [(runningLoss / len(dataloader)), runningMetrics]

        if not savePredictions is None:
            for k in predictions:
                predictions[k] = np.concatenate(predictions[k])

            #toReturn.append(predictions)
            if not os.path.exists(os.path.join(filesPath(conf), "predictions")):
                os.makedirs(os.path.join(filesPath(conf), "predictions"))
            np.savez(os.path.join(filesPath(conf), "predictions", "pred{}.npz".format(savePredictions)), **predictions)
            if not conf.has("nonVerbose") or conf.nonVerbose == False:
                print("Saved in {}".format(os.path.join(filesPath(conf), "predictions", "pred{}.npz".format(savePredictions))))

        if calculateHists:
            toReturn.append(hists)

        return toReturn

def runTrain(conf, model, optim, dataloaders):
    trainDataloader = dataloaders['train']
    validDataloader = dataloaders['valid']
    testDataloader = dataloaders['test']

    device = next(model.parameters()).device

    if conf.tensorBoard:
        writer = SummaryWriter(os.path.join(filesPath(conf),"tensorBoard"), flush_secs=60)


    if not conf.earlyStopping is None:
        maxMetric = 0.
        currPatience = 0.

    if not os.path.exists(os.path.join(filesPath(conf),"models")):
        os.makedirs(os.path.join(filesPath(conf),"models"))

    if not conf.bestSign in ['<', '>']:
        raise ValueError("bestSign {} not valid".format(conf.bestSign))
    if conf.bestSign == '>':
        bestValidMetric = - math.inf
    elif conf.bestSign == '<':
        bestValidMetric = math.inf

    for epoch in range(conf.startEpoch, conf.startEpoch+conf.epochs):
        startTime = datetime.now()
        if not conf.has("nonVerbose") or conf.nonVerbose == False:
            print("epoch {}".format(epoch), end='', flush=True)

        calculateHists = False
        if conf.tensorBoard and conf.has('histEveryEpochs'):
            if epoch % conf.histEveryEpochs == 0:
                calculateHists = True
                hists = model.getWeightsHists()

        model.train()
        for batchIndex, data in enumerate(trainDataloader):
            true = {k: data[k].to(device) for k in data}

            model.zero_grad()

            if calculateHists and (not conf.has('histOnlyBatches') or batchIndex < conf.histOnlyBatches): #only on first histOnlyBatches
                pred, currHists = model(true, returnHists=True)
                for h in currHists:
                    hh = h+"/train"
                    if not hh in hists:
                        hists[hh] = currHists[h]
                    else:
                        hists[hh] = np.concatenate((hists[hh], currHists[h]))
            else:
                pred = model(true)

            myLoss(pred, true, conf)
            optim.step()

        if not conf.has("nonVerbose") or conf.nonVerbose == False:
            print(": ", end='', flush=True)

        if calculateHists:
            for h in hists:
                writer.add_histogram(h, hists[h], epoch)

        trainLoss, trainMetrics = evaluate(conf, model, trainDataloader)
        if calculateHists:
            validLoss, validMetrics, validHists = evaluate(conf, model, validDataloader, calculateHists=True)
            for h in validHists:
                hh = h+"/valid"
                writer.add_histogram(hh, validHists[h], epoch)
        else:
            validLoss, validMetrics = evaluate(conf, model, validDataloader)
        testLoss, testMetrics = evaluate(conf, model, testDataloader)

        if conf.has("useTune") and conf.useTune:
            if not conf.bestKey in validMetrics.keys(): #conf.bestKey is used to control tune optimization
                raise ValueError("bestKey {} not present".format(conf.bestKey))
            tune.report(**{conf.bestKey: validMetrics[conf.bestKey].item()})

        writerDictLoss = {
            'train': trainLoss,
            'valid': validLoss,
            'test': testLoss,
            }

        writerDictMetrics = {}
        for k in trainMetrics: #same keys for train and valid
            writerDictMetrics[k] = {
                'train': trainMetrics[k],
                'valid': validMetrics[k],
                'test': testMetrics[k],
                }

        #save always last except when modelSave == "none"
        if conf.modelSave != "none":
            fileLast = os.path.join(filesPath(conf),"models","last.pt")

            if os.path.isfile(fileLast):
                os.remove(fileLast)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'epoch': epoch
            }, fileLast)

        if conf.modelSave == "all":
            fileModel = os.path.join(filesPath(conf), "models", "epoch{}.pt")
            torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'epoch': epoch
            }, fileModel.format(epoch))
        elif conf.modelSave == "best":
            fileModel = os.path.join(filesPath(conf), "models", "best.pt")
            if not conf.bestKey in validMetrics.keys():
                raise ValueError("bestKey {} not present".format(conf.bestKey))
            if (conf.bestSign == '<' and validMetrics[conf.bestKey] < bestValidMetric) or (conf.bestSign == '>' and validMetrics[conf.bestKey] > bestValidMetric):
                bestValidMetric = validMetrics[conf.bestKey]
                
                if os.path.isfile(fileModel):
                    os.remove(fileModel)

                torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'epoch': epoch
                }, fileModel)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if conf.tensorBoard:
                writer.add_scalars('loss', writerDictLoss, epoch)

                for k in writerDictMetrics:
                    writer.add_scalars('{}_{}'.format(conf.trackMetric, k), writerDictMetrics[k], epoch)

        endTime = datetime.now()
        if not conf.has("nonVerbose") or conf.nonVerbose == False:
            print("tr. loss {:0.3f}, tr. {} {:0.3f} - va. loss {:0.3f}, va. {} {:0.3f} - te. loss {:0.3f}, te. {} {:0.3f} ({}; exp. {})".format(trainLoss, conf.bestKey, trainMetrics[conf.bestKey], validLoss, conf.bestKey, validMetrics[conf.bestKey], testLoss, conf.bestKey, testMetrics[conf.bestKey], str(endTime-startTime).split('.', 2)[0], str((endTime-startTime)*(conf.startEpoch+conf.epochs-1-epoch)).split('.', 2)[0]), flush=True)
        
        if not conf.earlyStopping is None:
            if not conf.bestKey in validMetrics.keys():
                raise ValueError("bestKey {} not present".format(conf.bestKey))
            if not conf.bestSign in ['<', '>']:
                raise ValueError("bestSign {} not valid".format(conf.bestSign))
            #TODO: same of bestKey
            if (conf.bestSign == '<' and validMetrics[conf.bestKey] > maxMetric) or (conf.bestSign == '>' and validMetrics[conf.bestKey] < maxMetric):
                if currPatience >= conf.earlyStopping:
                    break
                currPatience += 1
            else:
                maxMetric = validMetrics[conf.bestKey]
                currPatience = 0
            
        #if not conf.earlyStopping is None:
        #    if validAcc < previousAccuracy:
        #        if currPatience >= conf.earlyStopping:
        #            break
        #        currPatience += 1
        #    else:
        #        currPatience = 0

        #if not conf.earlyStopping is None:
        #    if epoch >= conf.earlyStopping:
        #        if (validAcc <= previousAccuracies).all():
        #            break
        #    previousAccuracies[epoch%conf.earlyStopping] = validAcc

    time.sleep(120) #time to write tensorboard


def saveModel(conf, model, optim):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(), 
        }, os.path.join(filesPath(conf), "models", "model.pt"))

def loadModel(conf, device):
    fileToLoad = os.path.join(filesPath(conf), "models", conf.modelLoad)
    # if conf.modelSave == "best":
    #     fileToLoad = os.path.join(filesPath(conf), "models", "best.pt")
    # else:
    #     fileToLoad = os.path.join(filesPath(conf), "models", "epoch{}.pt".format(conf.loadEpoch))

    if not conf.has("nonVerbose") or conf.nonVerbose == False:
        print("Loading {}".format(fileToLoad), flush=True)

    model, optim = makeModel(conf, device)

    checkpoint = torch.load(fileToLoad, map_location=torch.device('cpu'))#map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optim.load_state_dict(checkpoint['optim_state_dict'])
    return model,optim,checkpoint['epoch']

def getLearningCurves(conf, metric='accuracy'):
    import tensorflow as tf
    
    sets = ['train', 'valid', 'test']

    path = {}
    for s in sets:
        path[s] = os.path.join(filesPath(conf), "tensorBoard", metric, s)

    logFiles = {}
    for s in sets:
        logFiles[s] = list(map(lambda f: os.path.join(path[s],f), sorted(os.listdir(path[s]))))

    values = {}
    for s in sets:
        values[s] = []
        for f in logFiles[s]:
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == 'loss' or v.tag == 'accuracy':
                        values[s].append(v.simple_value)

    return values

def getMaxValidEpoch(conf):
    values = getLearningCurves(conf)
    return np.argmax(values['valid'])

def getMaxValid(conf):
    try:
        values = getLearningCurves(conf)
        return values['valid'][np.argmax(values['valid'])]
    except (FileNotFoundError, ValueError):
        return -1

def getMaxTest(conf):
    try:
        values = getLearningCurves(conf)
        return values['test'][np.argmax(values['valid'])]
    except (FileNotFoundError, ValueError):
        return -1

def alreadyLaunched(conf):
    return os.path.isdir(os.path.join(filesPath(conf), "tensorBoard"))
