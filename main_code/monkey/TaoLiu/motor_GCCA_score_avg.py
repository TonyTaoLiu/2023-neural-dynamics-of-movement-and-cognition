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
GCCA_score_avg = []
for num_components in range(5,41,5): # change the number of components from 5 to 40 with step length 5
    GCCA_score = []
    num = 1
    for id1 in pairIndex_uni[0]:
        for id2 in pairIndex_uni[1]:
            for id3 in pairIndex_uni[2]:
                AllData = dt.get_data_array([allDFs[id1]] + [allDFs[id2]] + [allDFs[id3]], defs.exec_epoch,
                                            area=defs.areas[2], model=num_components) # perform pca
                data1 = np.reshape(AllData[0, ...], (-1, num_components))
                data2 = np.reshape(AllData[1, ...], (-1, num_components))
                data3 = np.reshape(AllData[2, ...], (-1, num_components))
                gcca = GCCA(latent_dims=num_components)
                gcca.fit([data1, data2, data3]) # align across 3 subjects
                GCCA_score.append(gcca.score([data1, data2, data3]))
                print('pair ' + str(num) + ' done')
                num += 1

    GCCA_score_avg.append(np.mean(np.array(GCCA_score)[:, :4], 1))

GCCA_score_avg = np.array(GCCA_score_avg)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
        'motor_MCx_aligned_GCCA_score_avg.npy',GCCA_score_avg)