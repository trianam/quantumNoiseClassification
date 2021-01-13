#     plotTrainingTune.py
#     To plot the learning curves.
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

from funPlot import plot
from itertools import product


# plot('config92', sets=['train','valid','test'], title="IID GRU t=1 bidirectional", colorsFirst=True)
# plot('config98', sets=['train','valid','test'], title="IID LSTM bidirectional t=1", colorsFirst=True)
# plot('config96', sets=['train','valid','test'], title="M1 GRU bidirectional t=0.1", colorsFirst=True)
# plot('config95', sets=['train','valid','test'], title="IID GRU bidirectional t=0.1", colorsFirst=True)
plot('config93', sets=['train','valid','test'], title="M1 GRU bidirectional t=1", colorsFirst=True)
