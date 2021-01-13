#     tuneConfigurations.py
#     The configurations for all the experiments.
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

from conf import Conf
from ray import tune
import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope

configGlobal = Conf({
    "useTune":              True,
    "nonVerbose":           True,
    "optimizer":            'adam',
    "learningRate":         0.001,
    "dimP":                 40,
    "dimPg":                2,
    "numT":                 15,
    "activation":           'relu',
    "normalizeP":           True,
    "batchSize":            16,
    "startEpoch":           0,
    "epochs":               100,
    "loss":                 "cce",
    "trackMetric":          "acc",
    "earlyStopping":        None,                          #None or patience
    "tensorBoard":          True,
    "histEveryEpochs":      10,
    "histOnlyBatches":      2,
    "modelSave":            "none",
    "bestKey":              "acc",
    "bestSign":             ">",
    "modelLoad":            'last.pt',
})


# ================================================ t=1 IID GRU bidirectional TEST
config92a = configGlobal.copy({
    "path":                 'config92a',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "activation":           'relu',
    "normalizeP":           False,
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "learningRate":         hp.choice("learningRate", [0.0001, 0.001, 0.01]),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "batchSize":            hp.choice("batchSize", [8, 16, 32]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),#scope.int((hp.qlognormal("hiddenDim", np.log(4), np.log(512), 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})






# ================================================ t=0.1 IID MLP
config71c = configGlobal.copy({
    "path":                 'config71c',
    "model":                'modelMLP',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})

# ================================================ t=0.1 IID MLPt
config71ct = configGlobal.copy({
    "path":                 'config71ct',
    "model":                'modelMLP',
    "allPt":                True,
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})

# ================================================ t=0.1 IID GRU
config79 = configGlobal.copy({
    "path":                 'config79',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'GRU',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IID GRU bidirectional
config95 = configGlobal.copy({
    "path":                 'config95',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IID GRU bidirectional max
config107 = configGlobal.copy({
    "path":                 'config107',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IID GRU bidirectional attention
config119 = configGlobal.copy({
    "path":                 'config119',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'attention',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})

# ================================================ t=0.1 IID LSTM
config89 = configGlobal.copy({
    "path":                 'config89',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'LSTM',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IID LSTM bidirectional
config101 = configGlobal.copy({
    "path":                 'config101',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IID LSTM bidirectional max
config113 = configGlobal.copy({
    "path":                 'config113',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'max',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IID LSTM bidirectional attention
config125 = configGlobal.copy({
    "path":                 'config125',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'attention',
    "dataset":              'datasetIIDb-scratch-t0.1/rep01.npz',
    "split":                'splitIIDb-scratch-t0.1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})




# ================================================ t=0.1 M1 MLP
config80 = configGlobal.copy({
    "path":                 'config80',
    "model":                'modelMLP',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})

# ================================================ t=0.1 M1 MLPt
config80t = configGlobal.copy({
    "path":                 'config80t',
    "model":                'modelMLP',
    "allPt":                True,
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})


# ================================================ t=0.1 M1 GRU
config81 = configGlobal.copy({
    "path":                 'config81',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'GRU',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})
config81b = configGlobal.copy({
    "path":                 'config81b',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'GRU',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
    }
})

# ================================================ t=0.1 M1 GRU bidirectional
config96 = configGlobal.copy({
    "path":                 'config96',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

config96b = configGlobal.copy({
    "path":                 'config96b',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
    }
})


# ================================================ t=0.1 M1 GRU bidirectional max
config108 = configGlobal.copy({
    "path":                 'config108',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

config108b = configGlobal.copy({
    "path":                 'config108b',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
    }
})

# ================================================ t=0.1 M1 GRU bidirectional attention
config120 = configGlobal.copy({
    "path":                 'config120',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'attention',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})

config120b = configGlobal.copy({
    "path":                 'config120b',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'attention',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})

# ================================================ t=0.1 M1 LSTM
config87 = configGlobal.copy({
    "path":                 'config87',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'LSTM',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})
config87b = configGlobal.copy({
    "path":                 'config87b',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'LSTM',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
    }
})

# ================================================ t=0.1 M1 LSTM bidirectional
config102 = configGlobal.copy({
    "path":                 'config102',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

config102b = configGlobal.copy({
    "path":                 'config102b',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
    }
})

# ================================================ t=0.1 M1 LSTM bidirectional max
config114 = configGlobal.copy({
    "path":                 'config114',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

config114b = configGlobal.copy({
    "path":                 'config114b',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
    }
})

# ================================================ t=0.1 M1 LSTM bidirectional attention
config126 = configGlobal.copy({
    "path":                 'config126',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'attention',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})

config126b = configGlobal.copy({
    "path":                 'config126b',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'attention',
    "dataset":              'datasetM1-scratch-t0.1/rep01.npz',
    "split":                'datasetM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})



# ================================================ t=0.1 IIDvsM1 MLP
config82 = configGlobal.copy({
    "path":                 'config82',
    "model":                'modelMLP',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})

# ================================================ t=0.1 IIDvsM1 MLP
config82t = configGlobal.copy({
    "path":                 'config82t',
    "model":                'modelMLP',
    "allPt":                True,
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})




# ================================================ t=0.1 IIDvsM1 GRU
config83 = configGlobal.copy({
    "path":                 'config83',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'GRU',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IIDvsM1 GRU bidirectional
config97 = configGlobal.copy({
    "path":                 'config97',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IIDvsM1 GRU bidirectional max
config109 = configGlobal.copy({
    "path":                 'config109',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IIDvsM1 GRU bidirectional attention
config121 = configGlobal.copy({
    "path":                 'config121',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'attention',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})

# ================================================ t=0.1 IIDvsM1 LSTM
config88 = configGlobal.copy({
    "path":                 'config88',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'LSTM',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IIDvsM1 LSTM bidirectional
config103 = configGlobal.copy({
    "path":                 'config103',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IIDvsM1 LSTM bidirectional max
config115 = configGlobal.copy({
    "path":                 'config115',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'max',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=0.1 IIDvsM1 LSTM bidirectional attention
config127 = configGlobal.copy({
    "path":                 'config127',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'attention',
    "dataset":              'datasetIIDvsM1-scratch-t0.1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t0.1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})











# ================================================ t=1 IID MLP
config71 = configGlobal.copy({
    "path":                 'config71',
    "model":                'modelMLP',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})

# ================================================ t=1 IID MLPt
config71t = configGlobal.copy({
    "path":                 'config71t',
    "model":                'modelMLP',
    "allPt":                True,
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})



# ================================================ t=1 IID GRU
config74 = configGlobal.copy({
    "path":                 'config74',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'GRU',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IID GRU bidirectional
config92 = configGlobal.copy({
    "path":                 'config92',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IID GRU bidirectional max
config104 = configGlobal.copy({
    "path":                 'config104',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IID GRU bidirectional attention
config116 = configGlobal.copy({
    "path":                 'config116',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'attention',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})

# ================================================ t=1 IID LSTM
config86 = configGlobal.copy({
    "path":                 'config86',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'LSTM',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IID LSTM bidirectional
config98 = configGlobal.copy({
    "path":                 'config98',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IID LSTM bidirectional max
config110 = configGlobal.copy({
    "path":                 'config110',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'max',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IID LSTM bidirectional attention
config122 = configGlobal.copy({
    "path":                 'config122',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'attention',
    "dataset":              'datasetIIDb-scratch-t1/rep01.npz',
    "split":                'splitIIDb-scratch-t1.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})




# ================================================ t=1 M1 MLP
config75 = configGlobal.copy({
    "path":                 'config75',
    "model":                'modelMLP',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})

# ================================================ t=1 M1 MLPt
config75t = configGlobal.copy({
    "path":                 'config75t',
    "model":                'modelMLP',
    "allPt":                True,
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})


# ================================================ t=1 M1 GRU
config76 = configGlobal.copy({
    "path":                 'config76',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'GRU',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 M1 GRU bidirectional
config93 = configGlobal.copy({
    "path":                 'config93',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 M1 GRU bidirectional max
config105 = configGlobal.copy({
    "path":                 'config105',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})
config105b = configGlobal.copy({
    "path":                 'config105b',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 4, 6, 1))),
    }
})

# ================================================ t=1 M1 GRU bidirectional attention
config117 = configGlobal.copy({
    "path":                 'config117',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'attention',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})

# ================================================ t=1 M1 LSTM
config90 = configGlobal.copy({
    "path":                 'config90',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'LSTM',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 M1 LSTM bidirectional
config99 = configGlobal.copy({
    "path":                 'config99',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 M1 LSTM bidirectional max
config111 = configGlobal.copy({
    "path":                 'config111',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 M1 LSTM bidirectional attention
config123 = configGlobal.copy({
    "path":                 'config123',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'attention',
    "dataset":              'datasetM1-scratch-t1/rep01.npz',
    "split":                'datasetM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})




# ================================================ t=1 IIDvsM1 MLP
config77 = configGlobal.copy({
    "path":                 'config77',
    "model":                'modelMLP',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})

# ================================================ t=1 IIDvsM1 MLPt
config77t = configGlobal.copy({
    "path":                 'config77t',
    "model":                'modelMLP',
    "allPt":                True,
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "activation":           hp.choice("activation", ['relu', 'sigmoid', 'tanh']),
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 0, 4, 1))),
    }
})




# ================================================ t=1 IIDvsM1 GRU
config78 = configGlobal.copy({
    "path":                 'config78',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'GRU',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IIDvsM1 GRU bidirectional
config94 = configGlobal.copy({
    "path":                 'config94',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IIDvsM1 GRU bidirectional max
config106 = configGlobal.copy({
    "path":                 'config106',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IIDvsM1 GRU bidirectional attention
config118 = configGlobal.copy({
    "path":                 'config118',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'attention',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})

# ================================================ t=1 IIDvsM1 LSTM
config91 = configGlobal.copy({
    "path":                 'config91',
    "model":                'modelRNN',
    "bidirectional":        False,
    "rnnType":              'LSTM',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IIDvsM1 LSTM bidirectional
config100 = configGlobal.copy({
    "path":                 'config100',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IIDvsM1 LSTM bidirectional max
config112 = configGlobal.copy({
    "path":                 'config112',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'max',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 IIDvsM1 LSTM bidirectional attention
config124 = configGlobal.copy({
    "path":                 'config124',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'LSTM',
    "aggregation":          'attention',
    "dataset":              'datasetIIDvsM1-scratch-t1/rep01.npz',
    "split":                'datasetIIDvsM1-scratch-t1/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
        "attentionDim":         scope.int((hp.quniform("attentionDim", 1, 512, 1))),
    }
})














# ================================================ t=2 N=15 IID GRU bidirectional max
config106T2N15 = configGlobal.copy({
    "path":                 'config106T2N15',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetIIDb-scratch-t2N15/rep01.npz',
    "split":                'datasetIIDb-scratch-t2N15/split-rep01.npz',
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=2 N=30 IID GRU bidirectional max
config106T2N30 = configGlobal.copy({
    "path":                 'config106T2N30',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetIIDb-scratch-t2N30/rep01.npz',
    "split":                'datasetIIDb-scratch-t2N30/split-rep01.npz',
    "numT":                 30,
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})







# ================================================ t=1 N=30 M1 GRU bidirectional max
config105T1N30 = configGlobal.copy({
    "path":                 'config105T1N30',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t1N30/rep01.npz',
    "split":                'datasetM1-scratch-t1N30/split-rep01.npz',
    "numT":                 30,
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 N=60 M1 GRU bidirectional max
config105T1N60 = configGlobal.copy({
    "path":                 'config105T1N60',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t1N60/rep01.npz',
    "split":                'datasetM1-scratch-t1N60/split-rep01.npz',
    "numT":                 60,
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 N=90 M1 GRU bidirectional max
config105T1N90 = configGlobal.copy({
    "path":                 'config105T1N90',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t1N90/rep01.npz',
    "split":                'datasetM1-scratch-t1N90/split-rep01.npz',
    "numT":                 90,
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 N=120 M1 GRU bidirectional max
config105T1N120 = configGlobal.copy({
    "path":                 'config105T1N120',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t1N120/rep01.npz',
    "split":                'datasetM1-scratch-t1N120/split-rep01.npz',
    "numT":                 120,
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})

# ================================================ t=1 N=150 M1 GRU bidirectional max
config105T1N150 = configGlobal.copy({
    "path":                 'config105T1N150',
    "model":                'modelRNN',
    "bidirectional":        True,
    "rnnType":              'GRU',
    "aggregation":          'max',
    "dataset":              'datasetM1-scratch-t1N150/rep01.npz',
    "split":                'datasetM1-scratch-t1N150/split-rep01.npz',
    "numT":                 150,
    "tuneConf":             {
        "dropout":              hp.choice("dropout", [0, 0.2, 0.5]),
        "weightDecay":          hp.choice("weightDecay", [0, 0.0001, 0.001]),
        "hiddenDim":            scope.int((hp.quniform("hiddenDim", 1, 512, 1))),
        "hiddenLayers":         scope.int((hp.quniform("hiddenLayers", 1, 4, 1))),
    }
})
