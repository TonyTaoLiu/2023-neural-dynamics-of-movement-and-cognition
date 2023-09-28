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

pairIndex_uni = [[],[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'Chewie' in animal: pairIndex_uni[0].append(i)
    if 'Mihili' in animal: pairIndex_uni[1].append(i)
    if 'Jaco' in animal: pairIndex_uni[2].append(i)

# aligning
latent_dynamics = []
alltar_latent_dynamics = []
CCA_score = []
warnings.filterwarnings("ignore")
num = 1
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            # perform pca to data of executing epoch of center out task, M1 + pMd, 10 components
            # change the number of components for analysis of center-out task
            AllData = dt.get_data_array([allDFs[id1]] + [allDFs[id2]] + [allDFs[id3]], defs.exec_epoch, area=defs.areas[2], model=defs.n_components)
            data1 = np.reshape(AllData[0, ...], (-1, defs.n_components))
            data2 = np.reshape(AllData[1, ...], (-1, defs.n_components))
            data3 = np.reshape(AllData[2, ...], (-1, defs.n_components))
            A1, B1, r1, *_ = dt.canoncorr(data1, data2, fullReturn=True) # align across 2 subjects
            A2, B2, r2, *_ = dt.canoncorr(data1, data3, fullReturn=True)
            A3, B3, r3, *_ = dt.canoncorr(data2, data3, fullReturn=True)
            CCA_score.append((id1, id2, id3, [r1, r2, r3]))
            U0, s0, Vh0 = svd(A1, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U1, s1, Vh1 = svd(B1, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U2, s2, Vh2 = svd(A2, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U3, s3, Vh3 = svd(B2, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U4, s4, Vh4 = svd(A3, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U5, s5, Vh5 = svd(B3, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            ex0_all1 = AllData[0, ...] @ U0 @ Vh0 # project to latent space
            ex1_all1 = AllData[1, ...] @ U1 @ Vh1
            ex0_all2 = AllData[0, ...] @ U2 @ Vh2
            ex2_all1 = AllData[2, ...] @ U3 @ Vh3
            ex1_all2 = AllData[1, ...] @ U4 @ Vh4
            ex2_all2 = AllData[2, ...] @ U5 @ Vh5
            alltar_latent_dynamics.append(
                (id1, id2, id3, [[ex0_all1, ex1_all1], [ex0_all2, ex2_all1], [ex1_all2, ex2_all2]]))
            for tar in range(8):
                ex0 = np.mean(AllData[0,tar, ...], axis=0)
                ex1 = np.mean(AllData[1,tar, ...], axis=0)
                ex2 = np.mean(AllData[2,tar, ...], axis=0)
                ex0_1 = ex0 @ U0 @ Vh0
                ex1_1 = ex1 @ U1 @ Vh1
                ex0_2 = ex0 @ U2 @ Vh2
                ex2_1 = ex2 @ U3 @ Vh3
                ex1_2 = ex1 @ U4 @ Vh4
                ex2_2 = ex2 @ U5 @ Vh5
                latent_dynamics.append((id1, id2, id3, tar, [[ex0_1, ex1_1], [ex0_2, ex2_1], [ex1_2, ex2_2]]))
            print('pair '+str(num)+' done')
            num += 1
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_aligned_latent_dynamics_2subj.npy',latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_alltar_aligned_latent_dynamics_2subj.npy',alltar_latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_MCx_aligned_CCA_score_2subj.npy',CCA_score)

print('save successfully')