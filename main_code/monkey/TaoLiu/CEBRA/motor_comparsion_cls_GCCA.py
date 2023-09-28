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
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import Ridge, LinearRegression, SGDClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tools.lstm import *

def centralize(mat):
    m = np.mean(mat, axis=0)
    mat = mat - m
    return mat

classifier_model = GaussianNB
# classifier_model = RandomForestClassifier
classifier_params = {}

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
local_root = pathlib.Path('E:\MSc Project\monkey dataset\motor\monkeyMotorData') # change with your own loading path
for animal, session in full_list:
    path = local_root/animal/session
    allDFs.append(defs.prep_general(dt.load_pyal_data(path))) # preprocess raw data
warnings.filterwarnings("default")

# form groups for alignment
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
cls_scores_set = []
for ii in range(1):
    cls_scores = []
    num = 1
    for id1,id2,testList in pairIndex_uni:
        # change defs.n_components to 10 for center-out task
        AllData = dt.get_data_array([allDFs[id1]]+[allDFs[id2]]+[allDFs[testid] for testid in testList],
                                    defs.exec_epoch, area=defs.areas[2], model=defs.n_components) # perform pca
        # adding history -- there is no need for history
        # AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)

        AllData1 = AllData[0, ...]
        _, n_trial1, n_time1, n_comp = AllData1.shape

        X1 = AllData1.reshape((-1, n_comp))

        AllData2 = AllData[1, ...]
        X2 = AllData2.reshape((-1, n_comp))

        outdim_size = n_comp

        for testId, _ in enumerate(testList):
            scores = []
            AllData3 = AllData[testId + 2, ...]  # index-0 is for the training dataset above
            # resizing
            X3 = AllData3.reshape((-1, n_comp))
            gcca = GCCA(latent_dims=outdim_size)
            gcca.fit([X1,X2,X3]) # align across 3 subjects

            X1_aligned = X1 @ gcca.weights[0] # project to latent space
            X2_aligned = X2 @ gcca.weights[1]
            X3_aligned = X3 @ gcca.weights[2]
            X1_aligned = X1_aligned.reshape((-1, n_time1 * n_comp))
            X2_aligned = X2_aligned.reshape((-1, n_time1 * n_comp))
            X3_aligned = X3_aligned.reshape((-1, n_time1 * n_comp))

            AllTar1 = np.repeat(np.arange(defs.n_targets), n_trial1) # labels

            trial_index1 = np.arange(len(AllTar1))
            # to guarantee shuffled ids
            while ((all_id_sh := rng.permutation(trial_index1)) == trial_index1).all():
                continue
            trial_index1 = all_id_sh
            X1_train, Y1_train = X1_aligned[trial_index1, :], AllTar1[trial_index1]
            classifier2 = classifier_model(**classifier_params)
            classifier2.fit(X1_train, Y1_train) # train classifier

            rng.shuffle(trial_index1)
            X2_test, Y2_test = X2_aligned[trial_index1, :], AllTar1[trial_index1]
            X3_test, Y3_test = X3_aligned[trial_index1, :], AllTar1[trial_index1]

            # test the decoder
            scores.append(classifier2.score(X2_test, Y2_test)) # test classifier
            scores.append(classifier2.score(X3_test, Y3_test))
            cls_scores.append((id1,id2,testId,scores))
        print('Pair '+str(num)+' done')
        num += 1
    cls_scores_set.append(cls_scores)
    print('########## Repeat '+str(ii)+' done ##########')
warnings.filterwarnings("default")
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_cls_scores_set_CEBRA_compare_GCCA.npy',cls_scores_set)
file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_cls_scores_set_CEBRA_compare_GCCA.txt','w')
for fp in cls_scores_set:
    file.write(str(fp))
    file.write('\n')
file.close()