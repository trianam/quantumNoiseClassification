#     funPlot.py
#     Collect the function used to plot the learning curves.
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

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import math
import numpy as np
import sys
import tuneConfigurations
import warnings
from ray.tune import Analysis

plt.rcParams['figure.dpi'] = 200

def plot(configs, sets='test', save=False, colorsFirst=False, title="", limits=None):
    keyLoss = 'loss'

    lineStyles = ['solid', 'dashed', 'dotted', 'dashdot']
    lineColors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    if not type(configs) is list:
        configs = [configs]

    if not type(sets) is list:
        sets = [sets]

    if len(configs)*len(sets) > len(lineStyles)*len(lineColors):
        raise ValueError("Too many curves to plot, {} of max {}.".format(len(configs)*len(sets), len(lineStyles)*len(lineColors)))

    fig = plt.figure(figsize=(8,9))
    #gs = fig.add_gridspec(2,1)
    #axLoss = fig.add_subplot(gs[0, 0])
    #axAcc = fig.add_subplot(gs[1, 0])
    axLoss = fig.add_axes([0.07, 0.53, 0.6, 0.42])
    axAcc = fig.add_axes([0.07, 0.05, 0.6, 0.42])
    #fig, (axLoss, axAcc) = plt.subplots(2)
    metrics = []
    i = 0
    for set in sets:
        for config in configs:
            myConfig = getattr(sys.modules['tuneConfigurations'], config)
            metrics.append(myConfig.trackMetric)

            #use tune to pick best in run
            analysis = Analysis(join("tuneOutput", myConfig.path))
            mode = ("max" if myConfig.bestSign == '>' else "min")
            #print("best hyperparameters for {}: {}".format(config, analysis.get_best_config(metric=myConfig.bestKey, mode=mode)))
            tunePath = analysis.get_best_logdir(metric=myConfig.bestKey, mode=mode)
            expPath = join(tunePath,'files', 'tensorBoard')
            keyAcc = "{}_{}".format(metrics[-1], metrics[-1])

            # metrics.append(getattr(sys.modules['configurations'], config).trackMetric)
            # keyAcc = "{}_{}".format(metrics[-1], metrics[-1])
            # expPath = join('files', getattr(sys.modules['configurations'], config).path, 'tensorBoard')
            try:
                keys = [f for f in listdir(expPath) if not isfile(join(expPath, f))]
            except FileNotFoundError:
                warnings.warn("Configuration {} not present. Skipping.".format(config))
                continue

            points = {}
            for k in keys:
                eventPathPart = join(expPath, k, set)
                for runPath in sorted([f for f in listdir(eventPathPart) if isfile(join(eventPathPart, f))]):
                    eventPath = join(eventPathPart, runPath)
                    ea = event_accumulator.EventAccumulator(eventPath)
                    ea.Reload()
                    if not k in points:
                        points[k] = [[v.step for v in ea.Scalars(k)], [v.value for v in ea.Scalars(k)]]
                    else:
                        points[k][0].extend([v.step for v in ea.Scalars(k)])
                        points[k][1].extend([v.value for v in ea.Scalars(k)])

            if limits is None:
                valuesLoss = points[keyLoss]
                valuesAcc = points[keyAcc]
            else:
                valuesLoss = [points[keyLoss][i][limits[0]:limits[1]] for i in [0,1]]
                valuesAcc = [points[keyAcc][i][limits[0]:limits[1]] for i in [0,1]]

            linesLoss = axLoss.plot(valuesLoss[0], valuesLoss[1], label="{} {}".format(config, set))
            linesAcc = axAcc.plot(valuesAcc[0], valuesAcc[1], label="{} {}".format(config, set))
            if colorsFirst:
                linesLoss[0].set_color(lineColors[i%len(lineColors)])
                linesLoss[0].set_linestyle(lineStyles[(i//len(lineColors))%len(lineStyles)])
                linesAcc[0].set_color(lineColors[i%len(lineColors)])
                linesAcc[0].set_linestyle(lineStyles[(i//len(lineColors))%len(lineStyles)])
            else:
                linesLoss[0].set_linestyle(lineStyles[i%len(lineStyles)])
                linesLoss[0].set_color(lineColors[(i//len(lineStyles))%len(lineColors)])
                linesAcc[0].set_linestyle(lineStyles[i%len(lineStyles)])
                linesAcc[0].set_color(lineColors[(i//len(lineStyles))%len(lineColors)])

            i += 1

    # axLoss.legend(loc='upper left', bbox_to_anchor=(1, 1),
    #          ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)
    #axLoss.set_title(title)
    axLoss.set_xlabel("Epoch")
    axLoss.set_ylabel("Loss")

    # axLoss.set_xticks(np.arange(0, round(axLoss.get_xlim()[1])+10, 10))
    # axLoss.set_xticks(np.arange(round(axLoss.get_xlim()[0]), round(axLoss.get_xlim()[1])+1, 1), minor=True)
    # axLoss.set_yticks(np.arange(0, round(axLoss.get_ylim()[1])+0.05, 0.05))
    # axLoss.set_yticks(np.arange(0, round(axLoss.get_ylim()[1])+0.01, 0.01), minor=True)
    axLoss.grid(which='both')
    axLoss.grid(which='minor', alpha=0.2)
    axLoss.grid(which='major', alpha=0.5)

    # axAcc.legend(loc='upper left', bbox_to_anchor=(1, 1),
    #           ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)
    #axAcc.set_title(title)
    axAcc.set_xlabel("Epoch")
    axAcc.set_ylabel("Metric ({})".format(list(dict.fromkeys(metrics))))
    #axAcc.set_ylim(0.49,1.)

    #axAcc.set_xticks(np.arange(0, round(axAcc.get_xlim()[1])+10, 10))
    #axAcc.set_xticks(np.arange(round(axAcc.get_xlim()[0]), round(axAcc.get_xlim()[1])+1, 1), minor=True)
    axAcc.set_yticks(np.arange(0.5, 1.01, 0.1))
    axAcc.set_yticks(np.arange(0.49, 1.01, 0.01), minor=True)
    axAcc.grid(which='both')
    axAcc.grid(which='minor', alpha=0.2)
    axAcc.grid(which='major', alpha=0.5)

    fig.suptitle(title)

    handles, labels = axAcc.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.68, 0.95), loc=2, borderaxespad=0.)
    # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1),
    #              bbox_transform = plt.gcf().transFigure,
    #              ncol=math.ceil(len(configs)/20), fancybox=True, shadow=True)

    # plt.legend( handles, labels, loc = 'upper left', bbox_to_anchor = (0.9,-0.1,2,2),
    #         bbox_transform = plt.gcf().transFigure )

    # plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), bbox_transform=plt.gcf().transFigure)

    # fig.subplots_adjust(wspace=2, hspace=2,left=0,top=2,right=2,bottom=0)

    #fig.tight_layout()

    #fig.subplots_adjust(right=2)

    fig.show()

    if save:
        fig.savefig('img/plot.eps')#, bbox_inches = 'tight')#, pad_inches = 0)

    plt.close()


def printMetrics(configs, printAllConfigs=False):
    sets = ['train', 'valid', 'test']

    if not type(configs) is list:
        configs = [configs]

    bestConfig = None
    points = {}
    for config in configs:
        myConfig = getattr(sys.modules['tuneConfigurations'], config)
        metric = myConfig.trackMetric

        try:
            #use tune to pick best in run
            analysis = Analysis(join("tuneOutput", myConfig.path))
        except ValueError:
            warnings.warn("Configuration {} not present. Skipping.".format(config))
            continue
        mode = ("max" if myConfig.bestSign == '>' else "min")
        print("best hyperparameters for {}: {}".format(config, analysis.get_best_config(metric=myConfig.bestKey, mode=mode)))
        tunePath = analysis.get_best_logdir(metric=myConfig.bestKey, mode=mode)
        expPath = join(tunePath,'files', 'tensorBoard')
        keyAcc = "{}_{}".format(metric, metric)
        try:
            keys = [f for f in listdir(expPath) if not isfile(join(expPath, f))]
        except FileNotFoundError:
            warnings.warn("Configuration {} not present. Skipping.".format(config))
            continue

        points[config] = {}
        for set in sets:
            points[config][set] = {}
            for k in keys:
                eventPathPart = join(expPath, k, set)
                for runPath in sorted([f for f in listdir(eventPathPart) if isfile(join(eventPathPart, f))]):
                    eventPath = join(eventPathPart, runPath)
                    ea = event_accumulator.EventAccumulator(eventPath)
                    ea.Reload()
                    if not k in points[config][set]:
                        points[config][set][k] = [[v.step for v in ea.Scalars(k)], [v.value for v in ea.Scalars(k)]]
                    else:
                        points[config][set][k][0].extend([v.step for v in ea.Scalars(k)])
                        points[config][set][k][1].extend([v.value for v in ea.Scalars(k)])

        bestSign = myConfig.bestSign
        if bestSign == '>':
            bestI = np.argmax(points[config]['valid'][keyAcc][1]) #point where better metric
        else:
            bestI = np.argmin(points[config]['valid'][keyAcc][1]) #point where better metric

        thisConfig = {
            'name': config,
            'epoch': bestI+1, 
            'train': points[config]['train'][keyAcc][1][bestI], 
            'valid': points[config]['valid'][keyAcc][1][bestI], 
            'test': points[config]['test'][keyAcc][1][bestI],
        }
        if printAllConfigs:
            print("{} (epoch {}):\ttrain {:.3};\tvalid {:.3};\ttest {:.3}".format(thisConfig['name'], thisConfig['epoch'], thisConfig['train'], thisConfig['valid'], thisConfig['test']))
            
        if bestConfig is None or (bestSign == '>' and thisConfig['valid']>bestConfig['valid']) or (bestSign == '<' and thisConfig['valid']<bestConfig['valid']):
            bestConfig = thisConfig    

    if not bestConfig is None:
        print("BEST ==== {} (epoch {}):\ttrain {:.3};\tvalid {:.3};\ttest {:.3}".format(bestConfig['name'], bestConfig['epoch'], bestConfig['train'], bestConfig['valid'], bestConfig['test']))
