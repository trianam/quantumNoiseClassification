#!/usr/bin/env python
# coding: utf-8

#     createDatasetScratch.py
#     Create the datasets for the experiments.
#     Copyright (C) 2021  Stefano Martina (stefano.martina@unifi.it), Stefano Gherardini (gherardini@lens.unifi.it)
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

import numpy as np
import sys
import os
import math
import warnings
import time
import datetime

def sampling_white_g(sorted_vector_P_g,sorted_vector_couplings_g):
    random_number = np.random.random()

    if random_number <= sorted_vector_P_g[0]:
        sample = sorted_vector_couplings_g[0]
    elif (random_number > sorted_vector_P_g[0]) and (random_number <= sorted_vector_P_g[0]+sorted_vector_P_g[1]):
        sample = sorted_vector_couplings_g[1]
    elif (random_number > sorted_vector_P_g[0]+sorted_vector_P_g[1]) and (random_number <= sorted_vector_P_g[0]+sorted_vector_P_g[1]+sorted_vector_P_g[2]):
        sample = sorted_vector_couplings_g[2]
    elif (random_number > sorted_vector_P_g[0]+sorted_vector_P_g[1]+sorted_vector_P_g[2]) and (random_number <= sorted_vector_P_g[0]+sorted_vector_P_g[1]+sorted_vector_P_g[2]+sorted_vector_P_g[3]):
        sample = sorted_vector_couplings_g[3]
    else:
        sample = sorted_vector_couplings_g[4]

    return sample

def sampling_colored_g_firstOrder_MC(transition_matrix,vector_couplings_g,a_priori_sample_g):
    '''
    We here implement a first-order-Markov for D=5 different values of g
    :param transition_matrix:
    :param vector_couplings_g:
    :param a_priori_sample_g:
    :return:
    '''

    random_number = np.random.random()

    if a_priori_sample_g == vector_couplings_g[0]:
        if random_number <= transition_matrix[0,0]:
            sample = vector_couplings_g[0]
        elif (random_number > transition_matrix[0,0]) and (random_number <= transition_matrix[0,0] + transition_matrix[0,1]):
            sample = vector_couplings_g[1]
        elif (random_number > transition_matrix[0,0] + transition_matrix[0,1]) and (random_number <= transition_matrix[0,0] + transition_matrix[0,1] + transition_matrix[0,2]):
            sample = vector_couplings_g[2]
        elif (random_number > transition_matrix[0,0] + transition_matrix[0,1] + transition_matrix[0,2]) and (random_number <= transition_matrix[0,0] + transition_matrix[0,1] + transition_matrix[0,2] + transition_matrix[0,3]):
            sample = vector_couplings_g[3]
        else:
            sample = vector_couplings_g[4]

    elif a_priori_sample_g == vector_couplings_g[1]:
        if random_number <= transition_matrix[1,0]:
            sample = vector_couplings_g[0]
        elif (random_number > transition_matrix[1,0]) and (random_number <= transition_matrix[1,0] + transition_matrix[1,1]):
            sample = vector_couplings_g[1]
        elif (random_number > transition_matrix[1,0] + transition_matrix[1,1]) and (random_number <= transition_matrix[1,0] + transition_matrix[1,1] + transition_matrix[1,2]):
            sample = vector_couplings_g[2]
        elif (random_number > transition_matrix[1,0] + transition_matrix[1,1] + transition_matrix[1,2]) and (random_number <= transition_matrix[1,0] + transition_matrix[1,1] + transition_matrix[1,2] + transition_matrix[1,3]):
            sample = vector_couplings_g[3]
        else:
            sample = vector_couplings_g[4]

    elif a_priori_sample_g == vector_couplings_g[2]:
        if random_number <= transition_matrix[2,0]:
            sample = vector_couplings_g[0]
        elif (random_number > transition_matrix[2,0]) and (random_number <= transition_matrix[2,0] + transition_matrix[2,1]):
            sample = vector_couplings_g[1]
        elif (random_number > transition_matrix[2,0] + transition_matrix[2,1]) and (random_number <= transition_matrix[2,0] + transition_matrix[2,1] + transition_matrix[2,2]):
            sample = vector_couplings_g[2]
        elif (random_number > transition_matrix[2,0] + transition_matrix[2,1] + transition_matrix[2,2]) and (random_number <= transition_matrix[2,0] + transition_matrix[2,1] + transition_matrix[2,2] + transition_matrix[2,3]):
            sample = vector_couplings_g[3]
        else:
            sample = vector_couplings_g[4]

    elif a_priori_sample_g == vector_couplings_g[3]:
        if random_number <= transition_matrix[3,0]:
            sample = vector_couplings_g[0]
        elif (random_number > transition_matrix[3,0]) and (random_number <= transition_matrix[3,0] + transition_matrix[3,1]):
            sample = vector_couplings_g[1]
        elif (random_number > transition_matrix[3,0] + transition_matrix[3,1]) and (random_number <= transition_matrix[3,0] + transition_matrix[3,1] + transition_matrix[3,2]):
            sample = vector_couplings_g[2]
        elif (random_number > transition_matrix[3,0] + transition_matrix[3,1] + transition_matrix[3,2]) and (random_number <= transition_matrix[3,0] + transition_matrix[3,1] + transition_matrix[3,2] + transition_matrix[3,3]):
            sample = vector_couplings_g[3]
        else:
            sample = vector_couplings_g[4]

    else:
        if random_number <= transition_matrix[4,0]:
            sample = vector_couplings_g[0]
        elif (random_number > transition_matrix[4,0]) and (random_number <= transition_matrix[4,0] + transition_matrix[4,1]):
            sample = vector_couplings_g[1]
        elif (random_number > transition_matrix[4,0] + transition_matrix[4,1]) and (random_number <= transition_matrix[4,0] + transition_matrix[4,1] + transition_matrix[4,2]):
            sample = vector_couplings_g[2]
        elif (random_number > transition_matrix[4,0] + transition_matrix[4,1] + transition_matrix[4,2]) and (random_number <= transition_matrix[4,0] + transition_matrix[4,1] + transition_matrix[4,2] + transition_matrix[4,3]):
            sample = vector_couplings_g[3]
        else:
            sample = vector_couplings_g[4]

    return sample

def Runge_Kutta_increment(rho,H):
    return -(1j)*(H@rho - rho@H)

def createDataset(outputFile, noiseSet, dtype=np.float32, startSeed=0, updateNoise=True, dim = 40, N = 15, step = 4*10**(-3), T_fin = 30, montecarlo_test = 10**4):
    '''
    Create a dataset with specified parameters. The dataset is for binary classification of the noise.
    :param outputFile: where to save file
    :param dtype: np.float32 or np.float64
    :param startSeed: random seed
    :param updateNoise: if True update noise at begin of every N, if False only once
    :param dim: equivale nel nostro caso al numero di nodi
    :param N: number of noise changes over time
    :param step: Integration step
    :param T_fin: duration of the dynamics
    :param montecarlo_test: how many samples generate (for every noise type)
    :return: nothing
    '''
    if not dtype is np.float32 and not dtype is np.float64:
        raise ValueError("Dtype {} not valid".format(dtype))

    np.random.seed(startSeed)

    tau = T_fin/N # time interval between two consecutive noise changes
    samples_tau = math.floor(tau/step) # samples of system evolution between noise changes

    # Più grande è il valore di "dim" (già 100 dovrebbe essere più che
    # sufficiente) e minore è la probabilità che ci sia un nodo isolato della
    # rete.

    noiseType1 = noiseSet['type1']
    noiseType2 = noiseSet['type2']

    # Random interaction coupling
    if noiseType1 != 'None':
        vector_couplings_g1 = np.array(noiseSet['couplings_g1'], dtype=dtype) #vector_couplings_g
        vector_P_g1 = np.array(noiseSet['P_g1'], dtype=dtype)
    if noiseType2 != 'None':
        vector_couplings_g2 = np.array(noiseSet['couplings_g2'], dtype=dtype) #vector_couplings_g
        vector_P_g2 = np.array(noiseSet['P_g2'], dtype=dtype)

    if noiseType1 == 'M=1':
        transition_matrix1 = np.array(noiseSet['transition_matrix1'], dtype=dtype)
    if noiseType2 == 'M=1':
        transition_matrix2 = np.array(noiseSet['transition_matrix2'], dtype=dtype)

    #cell_rho_tot_evolution = np.zeros((montecarlo_test,N+1))

    survival_probability = np.zeros((montecarlo_test*2,N+1,dim), dtype=dtype)
    noiseClass = np.zeros((montecarlo_test*2), dtype=np.int)

    print("Started {}".format(time.strftime('%d/%m/%Y %H:%M:%S')))
    startTime = time.time()
    currTime = startTime
    for k_montecarlo in range(montecarlo_test):
        # np.random.seed(k_montecarlo)
        prevTime = currTime
        currTime = time.time()
        print("Processing sample {}/{} (el. {}; exp. {})      ".format(k_montecarlo+1, montecarlo_test, datetime.timedelta(seconds=round(currTime-startTime)), datetime.timedelta(seconds=round((currTime-prevTime)*(montecarlo_test-k_montecarlo)))), end="\r")
        # Creazione Hamiltoniana del sistema con topologia random
        H_0 = np.zeros((dim,dim), dtype=dtype)
        prob_link = 0.5

        for k in range(dim):
            for j in range(k+1,dim):
                if np.random.random() <= prob_link:
                    H_0[k,j] = 1
                    H_0[j,k] = 1

        H1 = H_0
        H2 = H_0

        # Random rho_0 a singola eccitazione
        rho_0 = np.zeros((dim,dim), dtype=np.complex64 if dtype is np.float32 else np.complex128)
        r = np.random.randint(dim)
        rho_0[r,r] = 1

        rho1 = rho_0
        rho2 = rho_0

        if noiseType1 == 'M=1':
            aux_noise_sample1 = sampling_white_g(vector_P_g1, vector_couplings_g1)
        if noiseType2 == 'M=1':
            aux_noise_sample2 = sampling_white_g(vector_P_g2, vector_couplings_g2)

        noiseClass[k_montecarlo*2] = 0
        noiseClass[k_montecarlo*2+1] = 1

        
        index_meas_count = 0
        # While-cycle for the computation of the system dynamics
        for indice_evolution in range(N*samples_tau):
            # If-cycle that allows for the noise change after a time tau
            if indice_evolution % samples_tau == 0:
                with warnings.catch_warnings(): #diagonal is real
                    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
                    aux_surv_p_meas1 = np.array(rho1.diagonal(), dtype=dtype)
                    aux_surv_p_meas2 = np.array(rho2.diagonal(), dtype=dtype)

                survival_probability[k_montecarlo*2,index_meas_count] = aux_surv_p_meas1
                survival_probability[k_montecarlo*2+1,index_meas_count] = aux_surv_p_meas2
                #cell_rho_tot_evolution[k_montecarlo,index_meas_count] = rho

                if index_meas_count == 0 or updateNoise: #update noise always first time, and other times if required
                    if noiseType1 == 'IID':
                        aux_noise_sample1 = sampling_white_g(vector_P_g1, vector_couplings_g1)
                    elif noiseType1 == 'M=1':
                        aux_noise_sample1 = sampling_colored_g_firstOrder_MC(transition_matrix1, vector_couplings_g1, aux_noise_sample1)
                    elif noiseType1 != 'None':
                        raise ValueError("NoiseType {} not valid".format(noiseType1))

                    if noiseType2 == 'IID':
                        aux_noise_sample2 = sampling_white_g(vector_P_g2, vector_couplings_g2)
                    elif noiseType2 == 'M=1':
                        aux_noise_sample2 = sampling_colored_g_firstOrder_MC(transition_matrix2, vector_couplings_g2, aux_noise_sample2)
                    elif noiseType2 != 'None':
                        raise ValueError("NoiseType {} not valid".format(noiseType2))

                    if noiseType1 != 'None':
                        H1 = aux_noise_sample1*H_0
                    if noiseType2 != 'None':
                        H2 = aux_noise_sample2*H_0

                index_meas_count += 1

            # Evaluate evolution of the system with the Runge-kutta method
            k_1 = Runge_Kutta_increment(rho1,H1)
            k_2 = Runge_Kutta_increment(rho1 + (step/2)*k_1,H1)
            k_3 = Runge_Kutta_increment(rho1 + (step/2)*k_2,H1)
            k_4 = Runge_Kutta_increment(rho1 + step*k_3,H1)

            rho1 = rho1 + ((k_1/6)+(k_2/3)+(k_3/3)+(k_4/6))*step

            k_1 = Runge_Kutta_increment(rho2,H2)
            k_2 = Runge_Kutta_increment(rho2 + (step/2)*k_1,H2)
            k_3 = Runge_Kutta_increment(rho2 + (step/2)*k_2,H2)
            k_4 = Runge_Kutta_increment(rho2 + step*k_3,H2)

            rho2 = rho2 + ((k_1/6)+(k_2/3)+(k_3/3)+(k_4/6))*step

            #rho_evolution_aux.append(rho)

        # save last survival
        with warnings.catch_warnings(): #diagonal is real
            warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
            aux_surv_p_meas1 = np.array(rho1.diagonal(), dtype=dtype)
            aux_surv_p_meas2 = np.array(rho2.diagonal(), dtype=dtype)

        survival_probability[k_montecarlo*2,index_meas_count] = aux_surv_p_meas1
        survival_probability[k_montecarlo*2+1,index_meas_count] = aux_surv_p_meas2
        
    print("")
    endTime = time.time()
    print("Total time {}".format(datetime.timedelta(seconds=round(endTime-startTime))))
    print("Ended {}".format(time.strftime('%d/%m/%Y %H:%M:%S')))

    args = {
        'P': survival_probability,
        #'Pg': noiseClass,
        'Pg0c': noiseClass,
    }

    if not os.path.exists(os.path.dirname(outputFile)):
        os.makedirs(os.path.dirname(outputFile))
    np.savez_compressed(outputFile, **args)
    print("Saved in {}".format(outputFile))

def i2rowColumn(i):
    '''
    Transforms the index i in [0,n] in the tuple (r,c) where r and c are the row and column index in [0,m]
    of the corresponding inferior triangular matrix exluding the diagonal.
    :param i: the progressive index in [0,n]
    :return: (r,c) row and column indices in [0,m]
    '''
    r = math.ceil((math.sqrt(8*i+9)-1)/2)
    c = int(i - (r**2 - r)/2)
    return (r,c)

def rowColumn2i(r,c):
    '''
    Transforms the indices r and c in [0,m] of an inferior triangular matrix excluding the diagonal to
    a corresponding progressive index i in [0,n].
    :param r: the row index in [0,m]
    :param c: the column index in [0,m]
    :return: i the progressive index in [0,n]
    '''
    return int((r**2 - r + 2*c)/2)

def array2H(arr, dim=None):
    '''
    transform an array to the correnspondent hamiltonian representation (symmetric matrix with diagonal 0)
    :param arr: the array
    :param dim: the dimension of the hamiltonian
    :return: the correspondent hamiltonian
    '''
    if dim is None:
        dim = math.ceil((math.sqrt(8*len(arr)+9)-1)/2) +1

    H = np.zeros((dim,dim), dtype=arr.dtype)
    for i in range(len(arr)):
        r,c = i2rowColumn(i)
        H[r,c] = arr[i]
        H[c,r] = arr[i]

    return H

def createDatasetH(outputFile, splitOutputFile, noiseSet, dtype=np.float32, startSeed=0, updateNoise=True, dim = 40, N = 15, step = 4*10**(-3), T_fin = 30, montecarlo_test=10**4, number_H_0=10):
    '''
    Create a dataset with specified parameters. The dataset is for binary classification of H_0.
    :param outputFile: where to save file
    :param dtype: np.float32 or np.float64
    :param startSeed: random seed
    :param updateNoise: if True update noise at begin of every N, if False only once
    :param dim: equivale nel nostro caso al numero di nodi
    :param N: number of noise changes over time
    :param step: Integration step
    :param T_fin: duration of the dynamics
    :param montecarlo_test: how many samples generate (for every noise type)
    :return: nothing
    '''
    if not dtype is np.float32 and not dtype is np.float64:
        raise ValueError("Dtype {} not valid".format(dtype))

    np.random.seed(startSeed)

    tau = T_fin/N # time interval between two consecutive noise changes
    samples_tau = math.floor(tau/step) # samples of system evolution between noise changes

    # Più grande è il valore di "dim" (già 100 dovrebbe essere più che
    # sufficiente) e minore è la probabilità che ci sia un nodo isolato della
    # rete.

    noiseType = noiseSet['type']

    # Random interaction coupling
    if noiseType != 'None':
        vector_couplings_g = np.array(noiseSet['couplings_g'], dtype=dtype) #vector_couplings_g
        vector_P_g = np.array(noiseSet['P_g'], dtype=dtype)

    if noiseType == 'M=1':
        transition_matrix = np.array(noiseSet['transition_matrix'], dtype=dtype)

    #cell_rho_tot_evolution = np.zeros((montecarlo_test,N+1))

    dim_H_triang = int(((dim-1) * dim) / 2)

    all_H_0 = np.zeros((number_H_0, dim_H_triang), dtype=dtype)

    for i in range(number_H_0):
        # Creazione Hamiltoniana del sistema con topologia random
        valid = False
        while not valid: #check if not already generated
            h = np.random.randint(0,2,dim_H_triang)
            valid = not (all_H_0 == h).all(axis=1).any()
        all_H_0[i] = h

    testLen = int(montecarlo_test / 100. * 20.)
    allIndex = np.array(range(montecarlo_test), dtype=int)
    index = {
        'test': allIndex[-testLen:],
        'valid': allIndex[-2 * testLen: -testLen],
        'train': allIndex[: -2 * testLen],
    }

    testLenH0 = int(number_H_0 / 100. * 20.)
    allIndexH0 = np.array(range(number_H_0), dtype=int)
    indexH0 = {
        'test': allIndexH0[-testLenH0:],
        'valid': allIndexH0[-2 * testLenH0: -testLenH0],
        'train': allIndexH0[: -2 * testLenH0],
    }


    survival_probability = np.zeros((montecarlo_test,N+1,dim), dtype=dtype)
    return_H_0 = np.zeros((montecarlo_test,dim_H_triang), dtype=dtype)

    print("Started {}".format(time.strftime('%d/%m/%Y %H:%M:%S')))
    startTime = time.time()
    currTime = startTime
    for k_montecarlo in range(montecarlo_test):
        # np.random.seed(k_montecarlo)
        prevTime = currTime
        currTime = time.time()
        print("Processing sample {}/{} (el. {}; exp. {})      ".format(k_montecarlo+1, montecarlo_test, datetime.timedelta(seconds=round(currTime-startTime)), datetime.timedelta(seconds=round((currTime-prevTime)*(montecarlo_test-k_montecarlo)))), end="\r")

        for k in index:
            if k_montecarlo in index[k]: #if current sample is in train, valid or test
                H_0_array = all_H_0[indexH0[k][k_montecarlo%len(indexH0[k])]] #take one of the H0 in the corresponding set (so there are different H0 for different sets)
                break

        return_H_0[k_montecarlo] = H_0_array
        H_0 = array2H(H_0_array, dim)

        H = H_0

        # Random rho_0 a singola eccitazione
        rho_0 = np.zeros((dim,dim), dtype=np.complex64 if dtype is np.float32 else np.complex128)
        r = np.random.randint(dim)
        rho_0[r,r] = 1

        rho = rho_0

        if noiseType == 'M=1':
            aux_noise_sample = sampling_white_g(vector_P_g, vector_couplings_g)


        index_meas_count = 0
        # While-cycle for the computation of the system dynamics
        for indice_evolution in range(N*samples_tau):
            # If-cycle that allows for the noise change after a time tau
            if indice_evolution % samples_tau == 0:
                with warnings.catch_warnings(): #diagonal is real
                    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
                    aux_surv_p_meas = np.array(rho.diagonal(), dtype=dtype)

                survival_probability[k_montecarlo,index_meas_count] = aux_surv_p_meas
                #cell_rho_tot_evolution[k_montecarlo,index_meas_count] = rho

                if index_meas_count == 0 or updateNoise: #update noise always first time, and other times if required
                    if noiseType == 'IID':
                        aux_noise_sample = sampling_white_g(vector_P_g, vector_couplings_g)
                    elif noiseType == 'M=1':
                        aux_noise_sample = sampling_colored_g_firstOrder_MC(transition_matrix, vector_couplings_g, aux_noise_sample)
                    elif noiseType != 'None':
                        raise ValueError("NoiseType {} not valid".format(noiseType))

                    if noiseType != 'None':
                        H = aux_noise_sample*H_0

                index_meas_count += 1

            # Evaluate evolution of the system with the Runge-kutta method
            k_1 = Runge_Kutta_increment(rho,H)
            k_2 = Runge_Kutta_increment(rho + (step/2)*k_1,H)
            k_3 = Runge_Kutta_increment(rho + (step/2)*k_2,H)
            k_4 = Runge_Kutta_increment(rho + step*k_3,H)

            rho = rho + ((k_1/6)+(k_2/3)+(k_3/3)+(k_4/6))*step

            #rho_evolution_aux.append(rho)

        # save last survival
        with warnings.catch_warnings(): #diagonal is real
            warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
            aux_surv_p_meas = np.array(rho.diagonal(), dtype=dtype)

        survival_probability[k_montecarlo,index_meas_count] = aux_surv_p_meas

    print("")
    endTime = time.time()
    print("Total time {}".format(datetime.timedelta(seconds=round(endTime-startTime))))
    print("Ended {}".format(time.strftime('%d/%m/%Y %H:%M:%S')))

    args = {
        'P': survival_probability,
        'H_0': return_H_0,
    }

    if not os.path.exists(os.path.dirname(outputFile)):
        os.makedirs(os.path.dirname(outputFile))
    np.savez_compressed(outputFile, **args)
    print("Saved dataset in {}".format(outputFile))

    np.savez_compressed(splitOutputFile, train=index['train'], valid=index['valid'], test=index['test'])
    print("Saved split in {}".format(splitOutputFile))


def mergeDataset(inputPath, outputFile):
    args = {}
    for f in sorted(os.listdir(inputPath)):
        print("Loading {}".format(os.path.join(inputPath, f)))
        d = np.load(os.path.join(inputPath, f))
        for k in d:
            if not k in args:
                args[k] = d[k]
            else:
                args[k] = np.concatenate((args[k], d[k]), axis=0)

    np.savez_compressed(outputFile, **args)
    print("Saved in {}".format(outputFile))

def correctKey(inputFile):
    '''
    For the first version of the script, mistaken Pg for Pg0c
    :param inputFile: File to load and overwrite corrected
    :return: nothing
    '''
    args = {}
    print("Loading {}".format(inputFile))
    d = np.load(inputFile)
    for k in d:
        if k == 'Pg':
            args['Pg0c'] = d[k]
        else:
            args[k] = d[k]

    os.remove(inputFile)
    np.savez_compressed(inputFile, **args)
    print("Saved in {}".format(inputFile))

def splitDataset(inputFile, outputFile, randomize=False):
    '''
    Create a split
    :param inputFile:
    :param outputFile:
    :param randomize:
    :return:
    '''
    np.random.seed(42)
    print("processing {}".format(inputFile))

    dataset = np.load(inputFile)
    dataLen = len(dataset['P'])
    testLen = int(dataLen / 100. * 20.)

    if randomize:
        perm = np.random.permutation(dataLen)
    else:
        perm = np.array(range(dataLen))

    test = perm[-testLen:]
    valid = perm[-2 * testLen: -testLen]
    train = perm[: -2 * testLen]

    np.savez_compressed(outputFile, train=train, valid=valid, test=test)
    print("Saved in {}".format(outputFile))

def main():
    noiseSets = {
        'None': {
            'type': 'None',
        },
        'IID': {
            'type1': 'IID',
            'type2': 'IID',
            'couplings_g1' : [1, 2, 3, 4, 5],
            'couplings_g2' : [6, 7, 8, 9, 10],
            'P_g1' : [0.0124,0.04236,0.0820,0.2398,0.6234],
            'P_g2' : [0.1782,0.1865,0.2,0.2107,0.2245],
        },
        'IIDb': {
            'type1': 'IID',
            'type2': 'IID',
            'couplings_g1' : [1, 2, 3, 4, 5],
            'couplings_g2' : [1, 2, 3, 4, 5],
            'P_g1' : [0.0124,0.04236,0.0820,0.2398,0.6234],
            'P_g2' : [0.1782,0.1865,0.2,0.2107,0.2245],
        },
        'M1': {
            'type1': 'M=1',
            'type2': 'M=1',
            'couplings_g1' : [1, 2, 3, 4, 5],
            'couplings_g2' : [1, 2, 3, 4, 5],
            'P_g1' : [0.0124,0.04236,0.0820,0.2398,0.6234],
            'P_g2' : [0.1782,0.1865,0.2,0.2107,0.2245],
            'transition_matrix1' :
                [
                    [0.1855, 0.1042, 0.1331, 0.3264, 0.2508],
                    [0.1087, 0.2569, 0.1809, 0.1699, 0.2836],
                    [0.1041, 0.2758, 0.2746, 0.1386, 0.2069],
                    [0.0320, 0.0227, 0.2236, 0.3282, 0.3935],
                    [0.0856, 0.3749, 0.3094, 0.0078, 0.2223]
                ],
            'transition_matrix2':
                [
                    [0.2786, 0.2732, 0.1442, 0.2410, 0.0629],
                    [0.6097, 0.0275, 0.2391, 0.0399, 0.0839],
                    [0.2920, 0.2464, 0.1124, 0.3370, 0.0122],
                    [0.1709, 0.1486, 0.2981, 0.3097, 0.0728],
                    [0.1608, 0.1463, 0.2122, 0.2329, 0.2478]
                ],
        },
        'IIDvsM1': {
            'type1': 'IID',
            'type2': 'M=1',
            'couplings_g1' : [1, 2, 3, 4, 5],
            'couplings_g2' : [1, 2, 3, 4, 5],
            'P_g1' : [0.0124,0.04236,0.0820,0.2398,0.6234],
            'P_g2' : [0.0124,0.04236,0.0820,0.2398,0.6234],
            'transition_matrix2' :
                [
                    [0.1855, 0.1042, 0.1331, 0.3264, 0.2508],
                    [0.1087, 0.2569, 0.1809, 0.1699, 0.2836],
                    [0.1041, 0.2758, 0.2746, 0.1386, 0.2069],
                    [0.0320, 0.0227, 0.2236, 0.3282, 0.3935],
                    [0.0856, 0.3749, 0.3094, 0.0078, 0.2223]
                ],
        },
        'IID1': {
            'type': 'IID',
            'couplings_g' : [1, 2, 3, 4, 5],
            'P_g' : [0.0124,0.04236,0.0820,0.2398,0.6234],
        },
    }
    
    menu = { #k: [text, {args}]
        # Dataset 10c
        '118': ["IIDb from scratch rep01 t0.1 fast", createDataset, {'outputFile': 'data/datasetIIDb-scratch-t0.1/rep01.npz', 'noiseSet': noiseSets['IIDb'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':0.1, 'step':0.0001}],
        's118': ["Split IIDb from scratch rep01 t0.1 fast", splitDataset, {'inputFile': 'data/datasetIIDb-scratch-t0.1/rep01.npz', 'outputFile': 'data/splitIIDb-scratch-t0.1.npz'}],
        
        # Dataset 10d
        '119': ["IIDb from scratch rep01 t1 fast", createDataset, {'outputFile': 'data/datasetIIDb-scratch-t1/rep01.npz', 'noiseSet': noiseSets['IIDb'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':1, 'step':0.001}],
        's119': ["Split IIDb from scratch rep01 t1 fast", splitDataset, {'inputFile': 'data/datasetIIDb-scratch-t1/rep01.npz', 'outputFile': 'data/splitIIDb-scratch-t1.npz'}],
        
        # Dataset 12
        '121': ["M=1 from scratch rep01 t1 fast", createDataset, {'outputFile': 'data/datasetM1-scratch-t1/rep01.npz', 'noiseSet': noiseSets['M1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':1, 'step':0.001}],
        's121': ["Split M=1 from scratch rep01 t1 fast", splitDataset, {'inputFile': 'data/datasetM1-scratch-t1/rep01.npz', 'outputFile': 'data/datasetM1-scratch-t1/split-rep01.npz'}],

        # Dataset 13
        '122': ["IID vs M=1 from scratch rep01 t1 fast", createDataset, {'outputFile': 'data/datasetIIDvsM1-scratch-t1/rep01.npz', 'noiseSet': noiseSets['IIDvsM1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':1, 'step':0.001}],
        's122': ["Split IID vs M=1 from scratch rep01 t1 fast", splitDataset, {'inputFile': 'data/datasetIIDvsM1-scratch-t1/rep01.npz', 'outputFile': 'data/datasetIIDvsM1-scratch-t1/split-rep01.npz'}],

        # Dataset 14
        '123': ["M=1 from scratch rep01 t0.1 fast", createDataset, {'outputFile': 'data/datasetM1-scratch-t0.1/rep01.npz', 'noiseSet': noiseSets['M1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':0.1, 'step':0.0001}],
        's123': ["Split M=1 from scratch rep01 t0.1 fast", splitDataset, {'inputFile': 'data/datasetM1-scratch-t0.1/rep01.npz', 'outputFile': 'data/datasetM1-scratch-t0.1/split-rep01.npz'}],

        # Dataset 15
        '124': ["IID vs M=1 from scratch rep01 t0.1 fast", createDataset, {'outputFile': 'data/datasetIIDvsM1-scratch-t0.1/rep01.npz', 'noiseSet': noiseSets['IIDvsM1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':0.1, 'step':0.0001}],
        's124': ["Split IID vs M=1 from scratch rep01 t0.1 fast", splitDataset, {'inputFile': 'data/datasetIIDvsM1-scratch-t0.1/rep01.npz', 'outputFile': 'data/datasetIIDvsM1-scratch-t0.1/split-rep01.npz'}],

        # Dataset with tFin 2 and 30 N
        '125': ["IIDb from scratch rep01 t2 N30 fast", createDataset, {'outputFile': 'data/datasetIIDb-scratch-t2N30/rep01.npz', 'noiseSet': noiseSets['IIDb'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':2, 'step':0.002, 'N':30}],
        's125': ["Split IIDb from scratch rep01 t2 N30 fast", splitDataset, {'inputFile': 'data/datasetIIDb-scratch-t2N30/rep01.npz', 'outputFile': 'data/datasetIIDb-scratch-t2N30/split-rep01.npz'}],

        # Dataset with tFin 2 and 15 N
        '126': ["IIDb from scratch rep01 t2 N15 fast", createDataset, {'outputFile': 'data/datasetIIDb-scratch-t2N15/rep01.npz', 'noiseSet': noiseSets['IIDb'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':2, 'step':0.002, 'N':15}],
        's126': ["Split IIDb from scratch rep01 t2 N15 fast", splitDataset, {'inputFile': 'data/datasetIIDb-scratch-t2N15/rep01.npz', 'outputFile': 'data/datasetIIDb-scratch-t2N15/split-rep01.npz'}],

        # Dataset with tFin 1 and 30 N
        '127': ["M=1 from scratch rep01 t1 N30 fast", createDataset, {'outputFile': 'data/datasetM1-scratch-t1N30/rep01.npz', 'noiseSet': noiseSets['M1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':1, 'step':0.002, 'N':30}],
        's127': ["Split M=1 from scratch rep01 t1 N30 fast", splitDataset, {'inputFile': 'data/datasetM1-scratch-t1N30/rep01.npz', 'outputFile': 'data/datasetM1-scratch-t1N30/split-rep01.npz'}],

        # Dataset with tFin 1 and 60 N
        '128': ["M=1 from scratch rep01 t1 N60 fast", createDataset, {'outputFile': 'data/datasetM1-scratch-t1N60/rep01.npz', 'noiseSet': noiseSets['M1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':1, 'step':0.002, 'N':60}],
        's128': ["Split M=1 from scratch rep01 t1 N60 fast", splitDataset, {'inputFile': 'data/datasetM1-scratch-t1N60/rep01.npz', 'outputFile': 'data/datasetM1-scratch-t1N60/split-rep01.npz'}],

        # Dataset with tFin 1 and 150 N
        '129': ["M=1 from scratch rep01 t1 N150 fast", createDataset, {'outputFile': 'data/datasetM1-scratch-t1N150/rep01.npz', 'noiseSet': noiseSets['M1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':1, 'step':0.002, 'N':150}],
        's129': ["Split M=1 from scratch rep01 t1 N150 fast", splitDataset, {'inputFile': 'data/datasetM1-scratch-t1N150/rep01.npz', 'outputFile': 'data/datasetM1-scratch-t1N150/split-rep01.npz'}],

        # Dataset with tFin 1 and 90 N
        '130': ["M=1 from scratch rep01 t1 N90 fast", createDataset, {'outputFile': 'data/datasetM1-scratch-t1N90/rep01.npz', 'noiseSet': noiseSets['M1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':1, 'step':0.002, 'N':90}],
        's130': ["Split M=1 from scratch rep01 t1 N90 fast", splitDataset, {'inputFile': 'data/datasetM1-scratch-t1N90/rep01.npz', 'outputFile': 'data/datasetM1-scratch-t1N90/split-rep01.npz'}],

        # Dataset with tFin 1 and 120 N
        '131': ["M=1 from scratch rep01 t1 N120 fast", createDataset, {'outputFile': 'data/datasetM1-scratch-t1N120/rep01.npz', 'noiseSet': noiseSets['M1'], 'dtype':np.float32, 'startSeed':1, 'updateNoise':True, 'T_fin':1, 'step':0.002, 'N':120}],
        's131': ["Split M=1 from scratch rep01 t1 N120 fast", splitDataset, {'inputFile': 'data/datasetM1-scratch-t1N120/rep01.npz', 'outputFile': 'data/datasetM1-scratch-t1N120/split-rep01.npz'}],

    }

    if len(sys.argv) < 2:
        for k in menu:
            print("{}) {}".format(k, menu[k][0]))
        n = input("Enter number (also from arg): ")
    else:
        n = sys.argv[1]

    try:
        print("Running {}".format(n))
        menu[n][1](**menu[n][2])
    except KeyError:
        print("Number not valid")

if __name__ == "__main__":
    main()
