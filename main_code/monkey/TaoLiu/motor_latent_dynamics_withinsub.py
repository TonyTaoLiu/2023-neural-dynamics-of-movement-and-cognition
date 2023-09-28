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
# from sklearn.manifold import TSNE


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

# forming pairs for alignment
pairIndex_uni = []
for i, (animal1, session1) in enumerate(full_list):
    for j, (animal2, session2) in enumerate(full_list):
        if animal1 != animal2: continue
        if session1 == session2: continue
        pairIndex_uni.append((i, j, []))
        for l, (animal3, session3) in enumerate(full_list):
            if animal1 != animal3 or animal2 != animal3: continue
            if session1 == session3 or session2 == session3: continue
            pairIndex_uni[-1][2].append(l)


# aligning
latent_dynamics = []
alltar_latent_dynamics = []
GCCA_score = []
warnings.filterwarnings("ignore")
num = 1
for id1,id2,testList in pairIndex_uni:
    # perform pca to data of executing epoch of center out task, M1 + pMd, 10 components
    # change the number of components for analysis of center-out task
    AllData = dt.get_data_array([allDFs[id1]] + [allDFs[id2]] + [allDFs[testid] for testid in testList], defs.exec_epoch, area=defs.areas[2], model=defs.n_components)
    data1 = np.reshape(AllData[0, ...], (-1, defs.n_components))
    data2 = np.reshape(AllData[1, ...], (-1, defs.n_components))
    for testId, _ in enumerate(testList):
        data3 = np.reshape(AllData[testId + 2, ...], (-1, defs.n_components))
        gcca = GCCA(latent_dims=defs.n_components)
        gcca.fit([data1, data2, data3]) # align across 3 sessions
        GCCA_score.append((id1, id2, testId+2, [gcca.score([data1, data2, data3])]))
        U0, s0, Vh0 = svd(gcca.weights[0], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
        U1, s1, Vh1 = svd(gcca.weights[1], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
        U2, s2, Vh2 = svd(gcca.weights[2], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
        ex0_all = AllData[0, ...] @ U0 @ Vh0 # project to latent space
        ex1_all = AllData[1, ...] @ U1 @ Vh1
        ex2_all = AllData[testId+2, ...] @ U2 @ Vh2
        alltar_latent_dynamics.append((id1, id2, testId+2, [ex0_all, ex1_all, ex2_all]))
        for tar in range(8):
            ex0 = np.mean(AllData[0,tar, ...], axis=0)
            ex1 = np.mean(AllData[1,tar, ...], axis=0)
            ex2 = np.mean(AllData[testId + 2,tar, ...], axis=0)
            ex0 = ex0 @ U0 @ Vh0
            ex1 = ex1 @ U1 @ Vh1
            ex2 = ex2 @ U2 @ Vh2
            latent_dynamics.append((id1,id2,testId+2,tar,[ex0,ex1,ex2]))
        print('pair '+str(num)+' done')
        num += 1
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_withinsub_latent_dynamics.npy',latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_alltar_withinsub_latent_dynamics.npy',alltar_latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_MCx_withinsub_GCCA_score.npy',GCCA_score)

file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_withinsub_latent_dynamics.txt','w')
for fp in latent_dynamics:
    file.write(str(fp))
    file.write('\n')
file.close()

file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_alltar_withinsub_latent_dynamics.txt','w')
for fp in alltar_latent_dynamics:
    file.write(str(fp))
    file.write('\n')
file.close()

file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_withinsub_GCCA_score.txt','w')
for fp in GCCA_score:
    file.write(str(fp))
    file.write('\n')
file.close()

print('save successfully')