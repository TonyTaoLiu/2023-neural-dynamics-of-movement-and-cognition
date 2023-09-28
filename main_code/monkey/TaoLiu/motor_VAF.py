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

pairIndex_uni = [[],[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'Chewie' in animal: pairIndex_uni[0].append(i)
    if 'Mihili' in animal: pairIndex_uni[1].append(i)
    if 'Jaco' in animal: pairIndex_uni[2].append(i)

warnings.filterwarnings("ignore")
GCCA_score = []
num = 1
AllVAF = []
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            data_list = [allDFs[id1]] + [allDFs[id2]] + [allDFs[id3]]
            area = defs.areas[2]
            model = PCA(n_components=20, svd_solver='full')
            VAF = np.empty((len(data_list), model.n_components))
            field = f'{area}_rates'

            rng = np.random.default_rng(12345)
            for session, df in enumerate(data_list):
                df_ = pyal.restrict_to_interval(df, epoch_fun=defs.exec_epoch)
                rates = np.concatenate(df_[field].values, axis=0)
                rates_model = model.fit(rates)
                VAF[session, :] = np.cumsum(rates_model.explained_variance_ratio_) # obtaining VAF
                df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, '_pca') # perform pca

            AllVAF.append(VAF)
            print('pair ' + str(num) + ' done')
            num += 1


# GCCA_score_avg = np.array(GCCA_score_avg)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
        'motor_MCx_VAF.npy',AllVAF)