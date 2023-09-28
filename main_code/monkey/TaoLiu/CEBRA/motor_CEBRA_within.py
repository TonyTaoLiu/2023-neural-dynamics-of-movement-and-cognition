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
import itertools


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
local_root = pathlib.Path('E:\MSc Project\monkey dataset\motor\monkeyMotorData') # change with your own loading path
for animal, session in full_list:
    path = local_root/animal/session
    allDFs.append(defs.prep_general(dt.load_pyal_data(path))) # preprocess raw data
warnings.filterwarnings("default")

print('load successfully')

# form session groups for alignment
pairIndex_uni = [[], [], [], []]
for i, (animal, session) in enumerate(full_list):
    if 'Chewie' in animal:
        if 'Chewie2' in animal:
            pairIndex_uni[1].append(i)
        else:
            pairIndex_uni[0].append(i)
    if 'Mihili' in animal: pairIndex_uni[2].append(i)
    if 'Jaco' in animal: pairIndex_uni[3].append(i)


input_labels = []
latent_dynamics_full = []
latent_dynamics_avg = []
max_iterations = 5000 # number of iterations
output_dim = 3 # dimension of latent embedding
for subj in pairIndex_uni:
    index_all = list(itertools.combinations(subj,3))
    for index in index_all:
        data_list = [allDFs[ii] for ii in index]
        # noting that defs.n_components should be changed to 10 for center-out task
        AllData, AllVel = defs.get_data_array_and_vel(data_list, defs.exec_epoch,
                                                      area=defs.areas[2], n_components=defs.n_components) # perform pca
        AllData_input = [AllData[sess,...].reshape([-1, AllData.shape[-1]]) for sess in range(AllData.shape[0])]
        AllVel_input = [AllVel[sess,...].reshape([-1, AllVel.shape[-1]]) for sess in range(AllVel.shape[0])]
        input_labels.append(AllVel)
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

        multi_cebra_model.fit(AllData_input, AllVel_input) # aligning across sessions

        for ii, X in enumerate(AllData_input):
            multi_embeddings.append(multi_cebra_model.transform(X, session_id=ii)) # project to latent space

        latent_dynamics_full.append((index, multi_embeddings))

        subj_embeddings = []
        for j in range(len(multi_embeddings)):
            trial_full = multi_embeddings[j].reshape((n_targets, AllData.shape[2], AllData.shape[3], output_dim))
            subj_embeddings.append(trial_full)
        latent_dynamics_avg.append((subj, subj_embeddings))

# change with your own saving path
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_latent_dynamics_within_pca10_full.pickle','wb') as f:
    pickle.dump(latent_dynamics_full, f)

with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_latent_dynamics_within_pca10_reshape.pickle', 'wb') as f:
    pickle.dump(latent_dynamics_avg, f)

with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_input_labels_within_pca10.pickle', 'wb') as f:
    pickle.dump(input_labels, f)