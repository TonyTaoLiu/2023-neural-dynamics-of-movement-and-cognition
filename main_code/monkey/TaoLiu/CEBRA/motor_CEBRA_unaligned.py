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
from monkey.TaoLiu.dataset_selection import GoodDataList
from monkey.defs import *
import pyaldata as pyal
from cebra import CEBRA


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

print('load successfully')

pairIndex_uni = [[], [], []]
for i, (animal, session) in enumerate(full_list):
    if 'Chewie' in animal: pairIndex_uni[0].append(i)
    if 'Mihili' in animal: pairIndex_uni[1].append(i)
    if 'Jaco' in animal: pairIndex_uni[2].append(i)

# align the number of trials of all input sessions (function 'get_data_array_and_vel' only allows aligning number of trials within the input group)
n_shared_trial = np.inf
for i, subj in enumerate(pairIndex_uni):
    data_list = [allDFs[ii] for ii in subj]
    field = f'{defs.areas[2]}_rates'
    target_ids = np.unique(data_list[0].target_id)
    for df in data_list:
        for target in target_ids:
            df_ = pyal.select_trials(df, df.target_id == target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial)) # align number of trials

    n_shared_trial = int(n_shared_trial)

input_labels = []
names = ['Chewie','Mihili','Jaco']
latent_dynamics_full = []
latent_dynamics_avg = []
max_iterations = 5000 # number of iterations
output_dim = 3 # dimension of latent embedding
for i,subj in enumerate(pairIndex_uni):
    data_list = [allDFs[ii] for ii in subj]

    field = f'{defs.areas[2]}_rates'

    # finding the number of timepoints
    df_ = pyal.restrict_to_interval(data_list[0],epoch_fun=defs.exec_epoch)
    n_timepoints = int(df_[field][0].shape[0])

    # pre-allocating the data matrix
    AllData = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_components))
    AllVel = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, 2))
    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=defs.exec_epoch)
        rates = np.concatenate(df_[field].values, axis=0)
        rates_model = PCA(n_components=n_components, svd_solver='full').fit(rates)
        df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, '_pca') # perform pca
        vel_mean = np.nanmean(pyal.concat_trials(df, 'pos'), axis=0)

        for target in range(n_targets):
            df__ = pyal.select_trials(df_, df_.target_id == target)
            all_id = df__.trial_id.to_numpy()
            rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, (trial_rates, trial_vel) in enumerate(zip(df__._pca, df__.pos)):
                AllData[session, target, trial, :, :] = trial_rates
                AllVel[session, target, trial, :, :] = trial_vel - vel_mean

    AllData_input = [AllData[sess, ...].reshape([-1, AllData.shape[-1]]) for sess in range(AllData.shape[0])]
    AllVel_input = [AllVel[sess, ...].reshape([-1, AllVel.shape[-1]]) for sess in range(AllVel.shape[0])]
    input_labels.append(AllVel)
    single_embeddings = []

    # single training
    for data_input,label_input in zip(AllData_input,AllVel_input):
        cebra_model = CEBRA(model_architecture='offset10-model',
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

        cebra_model.fit(data_input, label_input)
        single_embeddings.append(cebra_model.transform(data_input)) # project to latent space

    latent_dynamics_full.append((subj, single_embeddings))

    subj_embeddings = []
    for j in range(len(single_embeddings)):
        trial_full = single_embeddings[j].reshape((n_targets, n_shared_trial, n_timepoints, output_dim))
        subj_embeddings.append(trial_full)
    latent_dynamics_avg.append((subj,subj_embeddings))

# change with your own saving path
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/motor_latent_dynamics_unaligned_pca10_full_3d.pickle','wb') as f:
    pickle.dump(latent_dynamics_full, f)

with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/motor_latent_dynamics_unaligned_pca10_reshape_3d.pickle', 'wb') as f:
    pickle.dump(latent_dynamics_avg, f)

with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/motor_input_labels_unaligned_pca10_3d.pickle', 'wb') as f:
    pickle.dump(input_labels, f)