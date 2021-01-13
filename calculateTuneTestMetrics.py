#!/usr/bin/env python
# coding: utf-8

#     calculateTuneTestMetrics.py
#     Reports all the results after the training for the best models in the considered categories.
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

from funPlot import printMetrics
from itertools import product
import numpy as np

print("========================================================================================================== T=0.1")
print("------------------------------------------------------------------------------------IID MLP")
printMetrics('config71c', tune=True)

print("\n------------------------------------------------------------------------------------IID GRU")
printMetrics('config79', tune=True)

print("------------------------------------------------------------------------------------IID MLPt")
printMetrics('config71ct', tune=True)

print("------------------------------------------------------------------------------------IID GRU bidirectional")
printMetrics('config95', tune=True)

print("------------------------------------------------------------------------------------IID GRU bidirectional max")
printMetrics('config107', tune=True)

print("------------------------------------------------------------------------------------IID GRU bidirectional attention")
printMetrics('config119', tune=True)

print("\n------------------------------------------------------------------------------------IID LSTM")
printMetrics('config89', tune=True)

print("------------------------------------------------------------------------------------IID LSTM bidirectional")
printMetrics('config101', tune=True)

print("------------------------------------------------------------------------------------IID LSTM bidirectional max")
printMetrics('config113', tune=True)

print("------------------------------------------------------------------------------------IID LSTM bidirectional attention")
printMetrics('config125', tune=True)

print("\n------------------------------------------------------------------------------------M1 MLP")
printMetrics('config80', tune=True)

print("\n------------------------------------------------------------------------------------M1 GRU")
printMetrics('config81', tune=True)
printMetrics('config81b', tune=True)

print("\n------------------------------------------------------------------------------------M1 MLPt")
printMetrics('config80t', tune=True)

print("------------------------------------------------------------------------------------M1 GRU bidirectional")
printMetrics('config96', tune=True)
printMetrics('config96b', tune=True)

print("------------------------------------------------------------------------------------M1 GRU bidirectional max")
printMetrics('config108', tune=True)
printMetrics('config108b', tune=True)

print("------------------------------------------------------------------------------------M1 GRU bidirectional attention")
printMetrics('config120', tune=True)
printMetrics('config120b', tune=True)

print("\n------------------------------------------------------------------------------------M1 LSTM")
printMetrics('config87', tune=True)
printMetrics('config87b', tune=True)

print("------------------------------------------------------------------------------------M1 LSTM bidirectional")
printMetrics('config102', tune=True)
printMetrics('config102b', tune=True)

print("------------------------------------------------------------------------------------M1 LSTM bidirectional max")
printMetrics('config114', tune=True)
printMetrics('config114b', tune=True)

print("------------------------------------------------------------------------------------M1 LSTM bidirectional attention")
printMetrics('config126', tune=True)
printMetrics('config126b', tune=True)

print("\n------------------------------------------------------------------------------------IIDvsM1 MLP")
printMetrics('config82', tune=True)

print("\n------------------------------------------------------------------------------------IIDvsM1 GRU")
printMetrics('config83', tune=True)

print("\n------------------------------------------------------------------------------------IIDvsM1 MLPt")
printMetrics('config82t', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 GRU bidirectional")
printMetrics('config97', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 GRU bidirectional max")
printMetrics('config109', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 GRU bidirectional attention")
printMetrics('config121', tune=True)

print("\n------------------------------------------------------------------------------------IIDvsM1 LSTM")
printMetrics('config88', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 LSTM bidirectional")
printMetrics('config103', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 LSTM bidirectional max")
printMetrics('config115', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 LSTM bidirectional attention")
printMetrics('config127', tune=True)


print("\n\n========================================================================================================== T=1")
print("------------------------------------------------------------------------------------IID MLP")
printMetrics('config71', tune=True)

print("\n------------------------------------------------------------------------------------IID GRU")
printMetrics('config74', tune=True)

print("------------------------------------------------------------------------------------IID MLPt")
printMetrics('config71t', tune=True)

print("------------------------------------------------------------------------------------IID GRU bidirectional")
printMetrics('config92', tune=True)

print("------------------------------------------------------------------------------------IID GRU bidirectional max")
printMetrics('config104', tune=True)

print("------------------------------------------------------------------------------------IID GRU bidirectional attention")
printMetrics('config116', tune=True)

print("\n------------------------------------------------------------------------------------IID LSTM")
printMetrics('config86', tune=True)

print("------------------------------------------------------------------------------------IID LSTM bidirectional")
printMetrics('config98', tune=True)

print("------------------------------------------------------------------------------------IID LSTM bidirectional max")
printMetrics('config110', tune=True)

print("------------------------------------------------------------------------------------IID LSTM bidirectional attention")
printMetrics('config122', tune=True)

print("\n------------------------------------------------------------------------------------M1 MLP")
printMetrics('config75', tune=True)

print("\n------------------------------------------------------------------------------------M1 GRU")
printMetrics('config76', tune=True)

print("\n------------------------------------------------------------------------------------M1 MLPt")
printMetrics('config75t', tune=True)

print("------------------------------------------------------------------------------------M1 GRU bidirectional")
printMetrics('config93', tune=True)

print("------------------------------------------------------------------------------------M1 GRU bidirectional max")
printMetrics('config105', tune=True)
printMetrics('config105b', tune=True)

print("------------------------------------------------------------------------------------M1 GRU bidirectional attention")
printMetrics('config117', tune=True)

print("\n------------------------------------------------------------------------------------M1 LSTM")
printMetrics('config90', tune=True)

print("------------------------------------------------------------------------------------M1 LSTM bidirectional")
printMetrics('config99', tune=True)

print("------------------------------------------------------------------------------------M1 LSTM bidirectional max")
printMetrics('config111', tune=True)

print("------------------------------------------------------------------------------------M1 LSTM bidirectional attention")
printMetrics('config123', tune=True)

print("\n------------------------------------------------------------------------------------IIDvsM1 MLP")
printMetrics('config77', tune=True)

print("\n------------------------------------------------------------------------------------IIDvsM1 GRU")
printMetrics('config78', tune=True)

print("\n------------------------------------------------------------------------------------IIDvsM1 MLPt")
printMetrics('config77t', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 GRU bidirectional")
printMetrics('config94', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 GRU bidirectional max")
printMetrics('config106', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 GRU bidirectional attention")
printMetrics('config118', tune=True)

print("\n------------------------------------------------------------------------------------IIDvsM1 LSTM")
printMetrics('config91', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 LSTM bidirectional")
printMetrics('config100', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 LSTM bidirectional max")
printMetrics('config112', tune=True)

print("------------------------------------------------------------------------------------IIDvsM1 LSTM bidirectional attention")
printMetrics('config124', tune=True)

print("\n\n========================================================================================================== T=2 IID GRU bidirectional max")
print("------------------------------------------------------------------------------------N15")
printMetrics('config106T2N15', tune=True)
print("------------------------------------------------------------------------------------N30")
printMetrics('config106T2N30', tune=True)

print("\n\n========================================================================================================== T=1 N=30 M1 GRU bidirectional max")
printMetrics('config105T1N30', tune=True)

print("\n\n========================================================================================================== T=1 N=60 M1 GRU bidirectional max")
printMetrics('config105T1N60', tune=True)

print("\n\n========================================================================================================== T=1 N=90 M1 GRU bidirectional max")
printMetrics('config105T1N90', tune=True)

print("\n\n========================================================================================================== T=1 N=120 M1 GRU bidirectional max")
printMetrics('config105T1N120', tune=True)

print("\n\n========================================================================================================== T=1 N=150 M1 GRU bidirectional max")
printMetrics('config105T1N150', tune=True)