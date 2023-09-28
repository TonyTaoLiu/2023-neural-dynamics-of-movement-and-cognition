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

def multimat_correlation(mat):
    '''
    this function is for calculating the correlation between multiple matrices. the idea is to expand the matrix first, then
    compute the correlation between the expanded vectors. The vectors can be concated to a new matrix as the input of numpy.correlation
    input: mat, list of matrices
    output: corr_mat, matrix containing correlation coefficients between input matrices
            avg_corr, average value of all correlation coefficients
    '''
    mat_1d_tmp = []
    for i in range(len(mat)):
        tmp = mat[i].reshape(-1)
        mat_1d_tmp.append(tmp)
    mat_1d = np.array(mat_1d_tmp)
    corr_mat = np.abs(np.corrcoef(mat_1d))
    corr_expand = np.triu(corr_mat, k=1).reshape(-1)
    corr = corr_expand[np.nonzero(corr_expand)]
    avg_corr = np.mean(corr_expand[np.nonzero(corr_expand)])
    return corr_mat, corr, avg_corr

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
corr_score = []
warnings.filterwarnings("ignore")
num = 1
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            # perform pca to data of executing epoch of center out task, M1 + pMd, 10 components
            # change the number of components for analysis of center-out task
            AllData = dt.get_data_array([allDFs[id1]] + [allDFs[id2]] + [allDFs[id3]], defs.exec_epoch, area=defs.areas[2], model=defs.n_components)
            ex0_all = AllData[0, ...]
            ex1_all = AllData[1, ...]
            ex2_all = AllData[2, ...]
            alltar_latent_dynamics.append((id1, id2, id3, [ex0_all, ex1_all, ex2_all])) # collect latent dynamics without averaging across trials

            data1 = np.reshape(AllData[0, ...], (-1, defs.n_components))
            data2 = np.reshape(AllData[1, ...], (-1, defs.n_components))
            data3 = np.reshape(AllData[2, ...], (-1, defs.n_components))
            corr_score_temp = []
            for com_ord in range(defs.n_components):
                # calculate correlations between components of different subjects
                _, _, avg_temp = multimat_correlation([data1[:,com_ord],data2[:,com_ord],data3[:,com_ord]])
                corr_score_temp.append(avg_temp)
            corr_score.append((id1,id2,id3,np.array(corr_score_temp)))

            for tar in range(8):
                ex0 = np.mean(AllData[0,tar, ...], axis=0) # mean latent dynamics across trials
                ex1 = np.mean(AllData[1,tar, ...], axis=0)
                ex2 = np.mean(AllData[2,tar, ...], axis=0)
                latent_dynamics.append((id1,id2,id3,tar,[ex0,ex1,ex2]))
            print('pair '+str(num)+' done')
            num += 1
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_unaligned_latent_dynamics.npy',latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_alltar_unaligned_latent_dynamics.npy',alltar_latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_MCx_unaligned_corr_score.npy',corr_score)
file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_unaligned_latent_dynamics.txt','w')
for fp in latent_dynamics:
    file.write(str(fp))
    file.write('\n')
file.close()

file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_alltar_unaligned_latent_dynamics.txt','w')
for fp in alltar_latent_dynamics:
    file.write(str(fp))
    file.write('\n')
file.close()

file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_unaligned_corr_score.txt','w')
for fp in corr_score:
    file.write(str(fp))
    file.write('\n')
file.close()

print('save successfully')