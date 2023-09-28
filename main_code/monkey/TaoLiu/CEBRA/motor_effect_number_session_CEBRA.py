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
from cebra import CEBRA
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
local_root = pathlib.Path('E:\MSc Project\monkey dataset\motor\monkeyMotorData') # change with your own loading path
for animal, session in full_list:
    path = local_root/animal/session
    allDFs.append(defs.prep_general(dt.load_pyal_data(path))) # preprocess raw data
warnings.filterwarnings("default")

print('load successfully')

pairIndex_uni = [[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'Chewie2' in animal:
        pairIndex_uni[0].append(i)
    elif 'Chewie' in animal:
        pairIndex_uni[1].append(i)

rng = params.rng
warnings.filterwarnings("ignore")
cls_scores_set = []
subj_num = 4 # can be either 2/3/4
output_dim = 10
max_iterations = 5000
for num,subj_id in enumerate(pairIndex_uni):
    cls_scores = []
    group_list = list(combinations(subj_id,subj_num))
    for group in group_list:
        data_list = [allDFs[group[i]] for i in range(len(group))]
    # data_list = [allDFs[subj_id[0]]] + [allDFs[subj_id[1]]] + [allDFs[subj_id[2]]]
        AllData, AllVel = defs.get_data_array_and_vel(data_list,defs.exec_epoch,area=defs.areas[2], n_components=10) # perform pca

        _,n_targets, n_trial1, n_time1, n_comp = AllData.shape

        AllData_input = [AllData[sess, ...].reshape([-1, AllData.shape[-1]]) for sess in range(AllData.shape[0])]
        AllVel_input = [AllVel[sess, ...].reshape([-1, AllVel.shape[-1]]) for sess in range(AllVel.shape[0])]
        multi_embeddings = []

        # Multisession training
        multi_cebra_model = CEBRA(model_architecture='offset10-model',
                                  batch_size=512,
                                  learning_rate=1e-4,
                                  temperature=1,
                                  output_dimension=output_dim,
                                  max_iterations=max_iterations,
                                  distance='cosine',
                                  conditional='time_delta',
                                  device='cuda',
                                  verbose=True,
                                  time_offsets=1)

        multi_cebra_model.fit(AllData_input, AllVel_input) # align across sessions

        for ii, X in enumerate(AllData_input):
            multi_embeddings.append(multi_cebra_model.transform(X, session_id=ii)) # project to latent space

        AllDyna = np.array([multi_embeddings[ii].reshape((n_targets, n_trial1, n_time1, output_dim))
                            for ii in range(len(multi_embeddings))])

        scores = [[] for num in range(AllData.shape[0])]
        for subj in range(AllData.shape[0]):
            X_train_tmp = AllDyna[subj].reshape((-1, n_time1 * n_comp))
            AllTar_train = np.repeat(np.arange(8), n_trial1) # labels
            trial_index1 = np.arange(len(AllTar_train))
            while ((all_id_sh := rng.permutation(trial_index1)) == trial_index1).all():
                continue
            trial_index1 = all_id_sh
            X_train, Y_train = X_train_tmp[trial_index1, :], AllTar_train[trial_index1]

            classifier = classifier_model(**classifier_params)
            classifier.fit(X_train, Y_train) # train classifier

            index_all = np.arange(AllData.shape[0])
            index_tmp = np.delete(index_all, subj) # exclude session for training
            for iii in range(index_tmp.shape[0]): # test on rest sessions
                X_test0 = AllDyna[index_tmp[iii]]
                X_test0 = X_test0.reshape((-1, n_time1 * n_comp))
                AllTar1 = np.repeat(np.arange(8), n_trial1)
                rng.shuffle(trial_index1)
                X_test, Y_test = X_test0[trial_index1, :], AllTar1[trial_index1]
                scores[subj].append(classifier.score(X_test, Y_test)) # test classifier
        cls_scores.append(scores)
    cls_scores_set.append(cls_scores)
    print('Pair ' + str(num) + ' done')
warnings.filterwarnings("default")
# change with your own saving path
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
          'motor_MCx_cls_scores_CEBRA_4.pickle', 'wb') as f:
    pickle.dump(cls_scores_set, f)
