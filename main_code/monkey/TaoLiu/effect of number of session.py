import sys, os, pathlib
import mat73
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle
import pyaldata as pyal
from sklearn.decomposition import PCA
from scipy.linalg import svd, inv
from sklearn.svm import LinearSVC
from cca_zoo.models import GCCA
from itertools import combinations
from utils import *

classifier_model = LinearSVC
classifier_params = {'max_iter':20000}

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
    root = pathlib.Path(nbPath/"data")
finally:
    os.chdir(nbPath)


data_path = '../../../cognitive data/sixSession'
file_name = 'Extracted_spikes_target_drift_30.mat' # change file name for analysis of different periods
# change the start and end point of window when analyzing different periods
allDFs,full_list,n_shared_trial,target_num,n_timepoints = load_data(data_path,file_name,117,133) # delay: 84-100; cue: 50-66

pairIndex_uni = [[],[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'B' in animal: pairIndex_uni[0].append(i)
    if 'K' in animal: pairIndex_uni[1].append(i)
    if 'L' in animal: pairIndex_uni[2].append(i)

rng = params.rng
warnings.filterwarnings("ignore")
canon_scores = []
cls_scores_set = []
subj_num = 6 # can be either 2/3/4 ... 6
output_dim = 30
for num,subj_id in enumerate(pairIndex_uni):
    cls_scores = []
    canon_scores_tmp = []
    group_list = list(combinations(subj_id,subj_num))
    for group in group_list:
        data_list = [allDFs[group[i]] for i in range(len(group))]
        AllData = get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, 30) # perform pca

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
            for iii in range(index_tmp.shape[0]): # test on rest sessions
                X2_aligned = X_test_aligned[index_tmp[iii]]
                AllTar1 = np.repeat(np.arange(target_num), n_trial1)

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
    cls_scores_set.append(cls_scores)
    canon_scores.append(canon_scores_tmp)
    print('Pair ' + str(num) + ' done')
warnings.filterwarnings("default")
# change with your own saving path
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
          'cognitive_prefrontal8A_target_cls_scores_set_6_0.03_30_drift.pickle', 'wb') as f:
    pickle.dump(cls_scores_set, f)

with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
          'cognitive_prefrontal8A_target_canonical_scores_set_6_0.03_30_drift.pickle', 'wb') as f:
    pickle.dump(canon_scores, f)