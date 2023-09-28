import os, sys, pathlib
from pprint import pprint
import gc
import pickle
from importlib import reload
import logging, warnings
logging.basicConfig(level=logging.ERROR)
import torch
import pandas as pd
import numpy as np
import sklearn
from scipy.linalg import svd
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import CCA
import scipy.linalg as linalg
# from monkey.TaoLiu.TCCA import TCCA
# from monkey.TaoLiu.gcca.gcca import GCCA
# from monkey.TaoLiu.cca_maxvar import cca_maxvar_multiview
from cca_zoo.models import MCCA,GCCA,TCCA,KCCA,KGCCA,KTCCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
# %matplotlib inline
from monkey.TaoLiu.dataset_selection import GoodDataList
from monkey.defs import *
import pyaldata as pyal
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
import random
from tools.lstm import *

'''
This code is for regression test on center-out task, with LSTM as regressor
'''

def centralize(mat):
    m = np.mean(mat, axis=0)
    mat = mat - m
    return mat

try:
    nbPath = pathlib.Path.cwd()
    RepoPath = nbPath.parent
    os.chdir(RepoPath)

    from tools import utilityTools as utility
    from tools import dataTools as dt
    import params
    defs = params.monkey_defs

    set_rc =  params.set_rc_params
    set_rc()
    # root = params.root
    root = pathlib.Path(RepoPath/"data")
finally:
    os.chdir(nbPath)

#if "__file__" not in dir():
full_list = []
#for area in ('M1','PMd'): # change here for region of interest
area = 'M1'
for animal, sessionList in GoodDataList[area].items(): # should write one after the data arrives
    if 'Mr' in animal:
        continue  # to remove MrT
    full_list.append((animal,sessionList))
full_list = [(animal,session) for animal,sessions in full_list for session in set(sessions)]

# load the DFs
warnings.filterwarnings("ignore")
allDFs = []
local_root = pathlib.Path('E:\MSc Project\monkey dataset\motor\monkeyMotorData')
for animal, session in full_list:
    path = local_root/animal/session
    allDFs.append(defs.prep_general(dt.load_pyal_data(path))) # preprocess raw data
warnings.filterwarnings("default")

# form pairs for alignment
pairIndex_uni = []
for i, (animal1, session1) in enumerate(full_list):
    for j, (animal2, session2) in enumerate(full_list):
        if animal1 == animal2: continue
        if 'Chewie' in animal1 and 'Chewie' in animal2: continue
        pairIndex_uni.append((i, j, []))
        for l, (animal3, session3) in enumerate(full_list):
            if animal1 == animal3 or animal2 == animal3: continue
            if 'Chewie' in animal1 and 'Chewie' in animal3: continue
            if 'Chewie' in animal2 and 'Chewie' in animal3: continue
            pairIndex_uni[-1][2].append(l)

rng = params.rng
warnings.filterwarnings("ignore")
reg_scores_set = []
for ii in range(1):
    reg_scores_across = []
    num = 1
    for id1, id2, testList in pairIndex_uni:
        AllData, AllVel = defs.get_data_array_and_vel([allDFs[id1]], defs.exec_epoch, area=defs.areas[2], n_components=defs.n_components)
        # AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)
        AllData1 = AllData[0, ...]
        AllVel1 = AllVel[0, ...]
        X_ = AllData1.reshape((-1, AllData1.shape[-2], AllData1.shape[-1]))
        Y_train = AllVel1.reshape((-1, AllVel1.shape[-2], AllVel1.shape[-1]))
        model = LSTMDecoder(10, 2)
        model.fit(X_, Y_train)

        for testId in testList:
            AllData, AllVel = defs.get_data_array_and_vel([allDFs[id2]], defs.exec_epoch, area=defs.areas[2],
                                                          n_components=defs.n_components)
            # AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)
            AllData2 = AllData[0, ...]
            AllVel2 = AllVel[0, ...]

            AllData, AllVel = defs.get_data_array_and_vel([allDFs[testId]], defs.exec_epoch, area=defs.areas[2], n_components=defs.n_components)
            # AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)
            AllData3 = AllData[0, ...]
            AllVel3 = AllVel[0, ...]
            # size matching
            *_,n_trial,n_time,n_comp = np.min(np.array((AllData1.shape,AllData2.shape,AllData3.shape)),axis=0)
            X1 = AllData1[:, :n_trial, :n_time, :n_comp].reshape((-1, n_comp))
            X2 = AllData2[:, :n_trial, :n_time, :n_comp].reshape((-1, n_comp))
            X3 = AllData3[:, :n_trial, :n_time, :n_comp].reshape((-1, n_comp))
            AllVel2 = AllVel2[:, :n_trial, :n_time, :].reshape((-1,n_time, 2))
            AllVel3 = AllVel3[:, :n_trial, :n_time, :].reshape((-1,n_time, 2))

            # alignment the decoder
            gcca = GCCA(latent_dims=n_comp)
            gcca.fit([X1, X2, X3])
            X2_aligned = X2 @ gcca.weights[1] @ linalg.inv(gcca.weights[0])
            X3_aligned = X3 @ gcca.weights[2] @ linalg.inv(gcca.weights[0])
            X2_aligned = X2_aligned.reshape((-1, n_time, n_comp))
            X3_aligned = X3_aligned.reshape((-1, n_time, n_comp))
            pr2, lab2 = model.predict(X2_aligned, AllVel2)
            pr3, lab3 = model.predict(X3_aligned, AllVel3)
            x2_score,y2_score = defs.custom_r2_func(pr2,lab2)
            x3_score,y3_score = defs.custom_r2_func(pr3, lab3)
            reg_scores_across.append((id1, id2, testId, (x2_score, y2_score,x3_score,y3_score)))
        print('Pair ' + str(num) + ' done')
        num += 1
    reg_scores_set.append(reg_scores_across)
    print('########## Repeat ' + str(ii) + ' done ##########')
warnings.filterwarnings("default")
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_reg_scores_set_lstm_nohist.npy',reg_scores_set)
file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_reg_scores_set_lstm_nohist.txt','w')
for fp in reg_scores_set:
    file.write(str(fp))
    file.write('\n')
file.close()