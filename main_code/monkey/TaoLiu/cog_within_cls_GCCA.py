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


data_path = '../../../cognitive data/actualUsed'
file_name = 'Extracted_spikes_cue_drift_30.mat' # change file name for analysis of different periods
# change the start and end point of window when analyzing different periods
allDFs,full_list,n_shared_trial,target_num,n_timepoints = load_data(data_path,file_name,50,66)# delay: 84-100; target: 117-133

pairIndex_uni = [[],[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'B' in animal: pairIndex_uni[0].append(i)
    if 'K' in animal: pairIndex_uni[1].append(i)
    if 'L' in animal: pairIndex_uni[2].append(i)

rng = params.rng
warnings.filterwarnings("ignore")
latent_dynamics_full = []
cls_scores = []
output_dim = 30
for num,subj_id in enumerate(pairIndex_uni):
    data_list = [allDFs[subj_id[0]]] + [allDFs[subj_id[1]]] + [allDFs[subj_id[2]]]
    AllData = get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, 30)

    _,n_targets, n_trial1, n_time1, n_comp = AllData.shape

    X1 = AllData[0].reshape([-1, AllData[0].shape[-1]])
    X2 = AllData[1].reshape([-1, AllData[1].shape[-1]])
    X3 = AllData[2].reshape([-1, AllData[2].shape[-1]])

    gcca = GCCA(latent_dims=output_dim)
    gcca.fit([X1, X2, X3]) # align across 3 subjects

    X1_test_aligned = X1 @ gcca.weights[0] # project to latent space
    X2_test_aligned = X2 @ gcca.weights[1]
    X3_test_aligned = X3 @ gcca.weights[2]
    X1_test_aligned = X1_test_aligned.reshape((-1, n_time1 * n_comp))
    X2_test_aligned = X2_test_aligned.reshape((-1, n_time1 * n_comp))
    X3_test_aligned = X3_test_aligned.reshape((-1, n_time1 * n_comp))
    X_test_aligned = [X1_test_aligned,X2_test_aligned,X3_test_aligned]

    scores = [[] for num in range(AllData.shape[0])]
    for subj in range(AllData.shape[0]):
        index_all = np.arange(AllData.shape[0])
        index_tmp = np.delete(index_all, subj) # exclude subject for training
        X1_aligned = X_test_aligned[subj]
        for iii in range(index_tmp.shape[0]): # test on rest subjects
            X2_aligned = X_test_aligned[index_tmp[iii]]
            AllTar1 = np.repeat(np.arange(target_num), n_trial1) # labels

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

    cls_scores.append((subj_id[0], subj_id[1], subj_id[2], scores))
    print('Pair ' + str(num) + ' done')
warnings.filterwarnings("default")
# change with your own saving path
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
          'cognitive_prefrontal8A_cue_cls_scores_set_within_0.03_30_drift.pickle', 'wb') as f:
    pickle.dump(cls_scores, f)

file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
          'cognitive_prefrontal8A_cue_cls_scores_set_within_0.03_30_drift.txt','w')
for fp in cls_scores:
    file.write(str(fp))
    file.write('\n')
file.close()