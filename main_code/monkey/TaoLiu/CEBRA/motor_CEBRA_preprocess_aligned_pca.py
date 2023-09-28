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
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import CCA
import scipy.linalg as linalg
# from monkey.TaoLiu.TCCA import TCCA
# from monkey.TaoLiu.gcca.gcca import GCCA
# from monkey.TaoLiu.cca_maxvar import cca_maxvar_multiview
from cca_zoo.models import MCCA,GCCA,TCCA,KCCA,KGCCA,KTCCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
# %matplotlib inline
from monkey.TaoLiu.dataset_selection import GoodDataList
from monkey.defs import *
import pyaldata as pyal
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
import random
from tools.lstm import *


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

    set_rc = params.set_rc_params
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

pairIndex_uni = [[],[],[]]
session_group = [[],[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'Chewie' in animal:
        pairIndex_uni[0].append(i)
        session_group[0].append(session)
    if 'Mihili' in animal:
        pairIndex_uni[1].append(i)
        session_group[1].append(session)
    if 'Jaco' in animal:
        pairIndex_uni[2].append(i)
        session_group[2].append(session)

np.save('E:\MSc Project\monkey dataset\motor/motor_CEBRA_pairIndex.npy',pairIndex_uni)
np.save('E:\MSc Project\monkey dataset\motor/motor_CEBRA_session_group.npy',session_group)

rng = params.rng
warnings.filterwarnings("ignore")
data_aligned = []
label_aligned = []
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            # noting that defs.n_components should be changed to 10 for center-out task
            AllData, AllVel = defs.get_data_array_and_vel([allDFs[id1]]+[allDFs[id2]]+[allDFs[id3]], defs.exec_epoch,
                                                          area=defs.areas[2], n_components=defs.n_components) # perform pca
            # AllData = dt.add_history_to_data_array(AllData, defs.MAX_HISTORY)
            data_aligned.append((id1, id2, id3, AllData))
            label_aligned.append((id1, id2, id3, AllVel))
# change with your own saving path
with open('E:\MSc Project\monkey dataset\motor/motor_CEBRA_data_aligned_pca10.pickle', 'wb') as f:
    pickle.dump(data_aligned, f)

with open('E:\MSc Project\monkey dataset\motor/motor_CEBRA_label_aligned_pca10.pickle', 'wb') as f:
    pickle.dump(label_aligned, f)