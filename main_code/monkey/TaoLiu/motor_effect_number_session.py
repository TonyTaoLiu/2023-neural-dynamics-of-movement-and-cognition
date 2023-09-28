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
import scipy.linalg as linalg
from cca_zoo.models import MCCA,GCCA,TCCA,KCCA,KGCCA,KTCCA
from monkey.TaoLiu.dataset_selection import GoodDataList
from monkey.defs import *
import pyaldata as pyal
from sklearn.naive_bayes import GaussianNB
from itertools import combinations


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

classifier_model = GaussianNB
classifier_params = {}

full_list = []
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

print('load successfully')

# only Chewie has more than 3 good sessions
pairIndex_uni = [[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'Chewie2' in animal:
        pairIndex_uni[0].append(i)
    elif 'Chewie' in animal:
        pairIndex_uni[1].append(i)

rng = params.rng
warnings.filterwarnings("ignore")
canon_scores = []
cls_scores_set = []
subj_num = 2 # can be either 2/3/4
output_dim = 10
for num,subj_id in enumerate(pairIndex_uni):
    cls_scores = []
    canon_scores_tmp = []
    group_list = list(combinations(subj_id,subj_num))
    for group in group_list:
        data_list = [allDFs[group[i]] for i in range(len(group))]
    # data_list = [allDFs[subj_id[0]]] + [allDFs[subj_id[1]]] + [allDFs[subj_id[2]]]
        AllData = dt.get_data_array(data_list,defs.exec_epoch,area=defs.areas[2], model=10) # perform pca

        _,n_targets, n_trial1, n_time1, n_comp = AllData.shape

        X = [AllData[i].reshape([-1, AllData[0].shape[-1]]) for i in range(len(group))]

        gcca = GCCA(latent_dims=output_dim)
        gcca.fit(X) # align across multiple sessions
        canon_scores_tmp.append(gcca.score(X))

        X_test_aligned = [X[i] @ gcca.weights[i] for i in range(len(X))]
        X_test_aligned = [X_test_aligned[i].reshape((-1, n_time1 * n_comp)) for i in range(len(X))]

        scores = [[] for num in range(AllData.shape[0])]
        for subj in range(AllData.shape[0]):
            index_all = np.arange(AllData.shape[0])
            index_tmp = np.delete(index_all, subj) # exclude session for training
            X1_aligned = X_test_aligned[subj]
            for iii in range(index_tmp.shape[0]): # test on the rest sessions
                X2_aligned = X_test_aligned[index_tmp[iii]]
                AllTar1 = np.repeat(np.arange(n_targets), n_trial1)

                trial_index1 = np.arange(len(AllTar1))
                # to guarantee shuffled ids
                while ((all_id_sh := rng.permutation(trial_index1)) == trial_index1).all():
                    continue
                trial_index1 = all_id_sh
                X1_train, Y1_train = X1_aligned[trial_index1, :], AllTar1[trial_index1]
                classifier1 = classifier_model(**classifier_params)
                classifier1.fit(X1_train, Y1_train) # train classifier
                rng.shuffle(trial_index1)
                X2_test, Y2_test = X2_aligned[trial_index1, :], AllTar1[trial_index1]
                scores[subj].append(classifier1.score(X2_test, Y2_test)) # test classifier
        cls_scores.append(scores)
    cls_scores_set.append(cls_scores) # collect classification accuracies of each group
    canon_scores.append(canon_scores_tmp) # collect canonical correlations of each group
    print('Pair ' + str(num) + ' done')
warnings.filterwarnings("default")
# change with your own saving path
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
          'motor_MCx_cls_scores_set_2.pickle', 'wb') as f:
    pickle.dump(cls_scores_set, f)

with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
          'motor_MCx_canonical_scores_set_2.pickle', 'wb') as f:
    pickle.dump(canon_scores, f)