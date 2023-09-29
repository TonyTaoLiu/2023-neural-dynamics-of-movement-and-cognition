import mat73
import os, sys, pathlib
import numpy as np
from monkey.defs import *
import pyaldata as pyal
import warnings
from scipy.stats import zscore
from sklearn.decomposition import PCA
from cca_zoo.models import GCCA
from scipy.linalg import svd, inv
from utils import *


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
allDFs,full_list,n_shared_trial,target_num,n_timepoints = load_data(data_path,file_name,50,66) # delay: 84-100; target: 117-133

pairIndex_uni = [[],[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'B' in animal: pairIndex_uni[0].append(i)
    if 'K' in animal: pairIndex_uni[1].append(i)
    if 'L' in animal: pairIndex_uni[2].append(i)

rng = params.rng
warnings.filterwarnings("ignore")
GCCA_score_avg = []
for num_components in range(5,41,5): # increasing number of components from 5 to 40, step length 5
    GCCA_score = []
    num = 1
    for id1 in pairIndex_uni[0]:
        for id2 in pairIndex_uni[1]:
            for id3 in pairIndex_uni[2]:
                data_list = [allDFs[id1]] + [allDFs[id2]] + [allDFs[id3]]
                AllData = get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, num_components) # performing pca
                data1 = np.reshape(AllData[0, ...], (-1, num_components))
                data2 = np.reshape(AllData[1, ...], (-1, num_components))
                data3 = np.reshape(AllData[2, ...], (-1, num_components))
                gcca = GCCA(latent_dims=num_components)
                gcca.fit([data1, data2, data3]) # aligning across 3 subjects
                GCCA_score.append(gcca.score([data1, data2, data3])) # obtaining gcca scores
                print('pair ' + str(num) + ' done')
                num += 1

    GCCA_score_avg.append(np.mean(np.array(GCCA_score)[:,:4],1))

GCCA_score_avg = np.array(GCCA_score_avg)
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
        'cog_prefrontal8A_cue_GCCA_score_avg_0.03_30_drift.npy',GCCA_score_avg)