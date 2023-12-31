import sys, os, pathlib
import mat73
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cebra import CEBRA
import warnings
import pickle
import pyaldata as pyal
from sklearn.decomposition import PCA
from scipy.linalg import svd, inv
from sklearn.svm import LinearSVC
from utils import *


classifier_model = LinearSVC
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
    root = pathlib.Path(nbPath/"data")
finally:
    os.chdir(nbPath)


data_path = '../../../../cognitive data/actualUsed'
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
latent_dynamics_avg = []
output_dim = 3 # dimension of latent dynamics
max_iterations = 1500 # number of iterations
num = 1
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            data_list = [allDFs[id1]] + [allDFs[id2]] + [allDFs[id3]]
            AllData = get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, defs.n_components) # perform pca
            train_size = int(AllData.shape[2]/5)*4 # 5-fold, 4 for train, 1 for test

            AllData_train = AllData[:,:,:train_size,:,:]
            _,n_targets, n_trial1, n_time1, _ = AllData_train.shape
            AllTar = np.repeat(np.arange(target_num), n_trial1)
            labels = np.repeat(AllTar, n_time1) + 0.0 # obtain train labels

            data1 = AllData_train[0].reshape([-1, AllData[0].shape[-1]])
            data2 = AllData_train[1].reshape([-1, AllData[1].shape[-1]])
            data3 = AllData_train[2].reshape([-1, AllData[2].shape[-1]])

            label1 = labels.reshape(-1)
            label2 = labels.reshape(-1)
            label3 = labels.reshape(-1)

            input_data = [data1, data2, data3]
            input_label = [label1, label2, label3]

            AllData_test = AllData[:, :, train_size:, :, :]
            _, _, n_trial2, _, _ = AllData_test.shape
            AllTar_test = np.repeat(np.arange(target_num), n_trial2)
            labels_test = np.repeat(AllTar_test, n_time1) + 0.0 # obtain test labels

            data1_test = AllData_test[0].reshape([-1, AllData[0].shape[-1]])
            data2_test = AllData_test[1].reshape([-1, AllData[1].shape[-1]])
            data3_test = AllData_test[2].reshape([-1, AllData[2].shape[-1]])

            label1_test = labels_test.reshape(-1)
            label2_test = labels_test.reshape(-1)
            label3_test = labels_test.reshape(-1)

            test_data = [data1_test, data2_test, data3_test]
            test_label = [label1_test, label2_test, label3_test]

            single_embeddings = []
            for xx, yy, xx_test in zip(input_data, input_label, test_data):
                # Multisession training
                cebra_model = CEBRA(model_architecture='offset10-model',
                                    batch_size=512,
                                    learning_rate=1e-4,
                                    temperature=1,
                                    output_dimension=output_dim,
                                    max_iterations=max_iterations,
                                    distance='cosine',
                                    conditional='time_delta',
                                    device='cuda_if_available',
                                    verbose=True,
                                    time_offsets=1)

                cebra_model.fit(xx, yy) # train on training set
                single_embeddings.append(cebra_model.transform(xx_test)) # transform test set

            latent_dynamics_full.append((id1,id2,id3,single_embeddings))
            # change with your own saving path
            with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result'
                      '\cognitive/cog_latent_dynamics_cue_unaligned_pca30_full_3d' + str(num) + '.pickle', 'wb') as f:
                pickle.dump(latent_dynamics_full, f)

            # divide based on different targets, reshape embeddings for visualization
            for i in range(target_num):
                target_embeddings = []
                for j in range(len(single_embeddings)):
                    target_trial = (test_label[j] == i)
                    # trial_avg = single_embeddings[j][target_trial, :].reshape(-1, n_time1, output_dim).mean(axis=0)
                    trial_avg = single_embeddings[j][target_trial, :].reshape(-1, n_time1, output_dim)
                    # trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
                    target_embeddings.append(trial_avg)
                latent_dynamics_avg.append((id1, id2, id3, i, target_embeddings))
            # change with your own saving path
            with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result'
                      '\cognitive/cog_latent_dynamics_cue_unaligned_pca30_full_reshape_3d' + str(num) + '.pickle', 'wb') as f:
                pickle.dump(latent_dynamics_avg, f)
            num = num + 1