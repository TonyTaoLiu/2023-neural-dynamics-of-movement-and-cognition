import mat73
import os, sys, pathlib
import numpy as np
from monkey.defs import *
import pyaldata as pyal
import warnings
from sklearn.decomposition import PCA
from scipy.linalg import svd
from utils import *

def multimat_correlation(mat):
    '''
    this function is for calculating the correlation between multiple matrices. the idea is to expand the matrix first, then
    compute the correlation between the expanded vectors. The vectors can be concated to a new matrix as the input of numpy.correlation
    input: mat, list of matrices
    output: corr_mat, matrix containing correlation coefficients between input matrices
            corr, correlation coefficients between matrices
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
    root = pathlib.Path(nbPath/"data")
finally:
    os.chdir(nbPath)

data_path = '../../../cognitive data/actualUsed'
file_name = 'Extracted_spikes_cue_drift_30.mat' # change file name for analysis of different periods
# change the start and end point of window when analyzing different periods
allDFs,full_list,n_shared_trial,target_num,n_timepoints = load_data(data_path,file_name,50,66)

print('load successfully')

pairIndex_uni = [[],[],[]]
for i, (animal, session) in enumerate(full_list):
    if 'B' in animal: pairIndex_uni[0].append(i)
    if 'K' in animal: pairIndex_uni[1].append(i)
    if 'L' in animal: pairIndex_uni[2].append(i)

rng = params.rng
latent_dynamics = []
alltar_latent_dynamics = []
corr_score = []
warnings.filterwarnings("ignore")
num = 1
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            data_list = [allDFs[id1]]+[allDFs[id2]]+[allDFs[id3]]
            AllData = get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, 30) # performing pca

            ex0_all = AllData[0, ...]
            ex1_all = AllData[1, ...]
            ex2_all = AllData[2, ...]
            alltar_latent_dynamics.append((id1, id2, id3, [ex0_all, ex1_all, ex2_all])) # obtain latent dynamics without averaging across trials

            data1 = np.reshape(AllData[0, ...], (-1, defs.n_components))
            data2 = np.reshape(AllData[1, ...], (-1, defs.n_components))
            data3 = np.reshape(AllData[2, ...], (-1, defs.n_components))
            corr_score_temp = []
            # compute correlations across principal components
            for com_ord in range(defs.n_components):
                _, _, avg_temp = multimat_correlation([data1[:, com_ord], data2[:, com_ord], data3[:, com_ord]])
                corr_score_temp.append(avg_temp)

            corr_score.append((id1, id2, id3, np.array(corr_score_temp)))

            for tar in range(AllData.shape[1]):
                ex0 = np.mean(AllData[0,tar, ...], axis=0) # mean latent dynamics across trials
                ex1 = np.mean(AllData[1,tar, ...], axis=0)
                ex2 = np.mean(AllData[2,tar, ...], axis=0)
                latent_dynamics.append((id1,id2,id3,tar,[ex0,ex1,ex2]))
            print('pair '+str(num)+' done')
            num += 1
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_latent_dynamics_unaligned_0.03_30_drift.npy',latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_alltar_cue_latent_dynamics_unaligned_0.03_30_drift.npy',alltar_latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_corr_score_unaligned_0.03_30_drift.npy',corr_score)

print('save successfully')