#!/usr/bin/env python
# coding: utf-8

#     tuneRunPytorch.py
#     The runner to launch one experiment.
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


#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
import socket
import tuneConfigurations
import funPytorch as fun
import notifier
from datetime import datetime
import time

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

timeFormat = "%Y/%m/%d - %H:%M:%S"

def createRunner(device, baseConf):
    def runner(confDict, checkpoint_dir=None):
        conf = baseConf.copy(confDict)

        if conf.startEpoch == 0:
            model,optim = fun.makeModel(conf, device)
        else:   #maybe this doesn't make sense here
            model,optim,_ = fun.loadModel(conf, device)
        dataloaders, _ = fun.processData(conf)
        fun.runTrain(conf, model, optim, dataloaders)

    return runner


if len(sys.argv) < 2 or len(sys.argv) > 5:
    print("Use {} configName [parProc (def. 10)] [grace (def. 1)] [gpuNum (def. 0)]".format(sys.argv[0]))
else:
    if len(sys.argv) >= 3:
        parProc = int(sys.argv[2])
    else:
        parProc = 10

    if len(sys.argv) >= 4:
        grace = int(sys.argv[3])
    else:
        grace = 1

    if len(sys.argv) >= 5:
        device = "cuda:{}".format(sys.argv[3])
    else:
        device = "cuda:0"

    conf = getattr(sys.modules['tuneConfigurations'], sys.argv[1])
    tuneConf = conf.tuneConf

    startTime = datetime.now()
    print("====================")
    print("RUN USING {} on device {}".format(sys.argv[1], device))
    print(startTime.strftime(timeFormat))
    print("====================")

    runner = createRunner(device, conf)

    mode = ("max" if conf.bestSign == '>' else "min")
    scheduler = ASHAScheduler(metric=conf.bestKey, mode=mode, grace_period=grace)#5)
    searchAlg = HyperOptSearch(tuneConf, metric=conf.bestKey, mode=mode)

    trialResources = {'cpu': 1., 'gpu': 1./parProc}

    analysis = None
    error = None
    try:
        analysis = tune.run(runner, config=tuneConf, name=conf.path, scheduler=scheduler, search_alg=searchAlg, resources_per_trial=trialResources, local_dir='tuneOutput', num_samples=1000)
    except Exception as e:
        error = e

    #recover with
    # from ray.tune import Analysis
    # analysis = Analysis("tuneOutput/"+conf.path)


    endTime = datetime.now()

    subject = "Tune Training of {} finished on {}".format(sys.argv[1], socket.gethostname())
    message = "=======\n" \
                "Start: {}\n" \
                "End:   {}\n" \
                "Duration: {}\n" \
                "====================\n".format(
        startTime.strftime(timeFormat),
        endTime.strftime(timeFormat),
        str(endTime-startTime).split('.', 2)[0]
    )

    if not analysis is None:
        message = message + "Best hyperparameters found were: {}\n" \
                "\n" \
                "Best logdir found were: {}\n" \
                "\n" \
                "Best last result found were: {}\n" \
                "====================".format(
            analysis.get_best_config(scope='all', metric=conf.bestKey, mode=mode),
            analysis.get_best_logdir(scope='all', metric=conf.bestKey, mode=mode),
            analysis.get_best_trial(scope='all', metric=conf.bestKey, mode=mode).last_result
        )

    if not error is None:
        subject = subject + " WITH ERRORS"
        message = message + "\nERROR!!!!\n===========\n{}".format(error)

    print(message)
    notifier.sendMessage(subject, message)
    time.sleep(15)
