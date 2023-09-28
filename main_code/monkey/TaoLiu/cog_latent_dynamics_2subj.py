import mat73
import os, sys, pathlib
import numpy as np
from monkey.defs import *
import pyaldata as pyal
import warnings
from sklearn.decomposition import PCA
from scipy.linalg import svd
from utils import *

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
CCA_score = []
warnings.filterwarnings("ignore")
num = 1
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            data_list = [allDFs[id1]]+[allDFs[id2]]+[allDFs[id3]]
            AllData = get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, 30) # performing pca

            data1 = np.reshape(AllData[0, ...], (-1, defs.n_components))
            data2 = np.reshape(AllData[1, ...], (-1, defs.n_components))
            data3 = np.reshape(AllData[2, ...], (-1, defs.n_components))
            A1,B1,r1,*_ = dt.canoncorr(data1,data2,fullReturn=True) # align across 2 subjects
            A2,B2,r2,*_ = dt.canoncorr(data1,data3,fullReturn=True)
            A3,B3,r3,*_ = dt.canoncorr(data2,data3,fullReturn=True)
            CCA_score.append((id1,id2,id3,[r1,r2,r3]))
            U0, s0, Vh0 = svd(A1, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U1, s1, Vh1 = svd(B1, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U2, s2, Vh2 = svd(A2, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U3, s3, Vh3 = svd(B2, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U4, s4, Vh4 = svd(A3, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U5, s5, Vh5 = svd(B3, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            ex0_all1 = AllData[0,...] @ U0 @ Vh0 # project to latent space
            ex1_all1 = AllData[1,...] @ U1 @ Vh1
            ex0_all2 = AllData[0,...] @ U2 @ Vh2
            ex2_all1 = AllData[2,...] @ U3 @ Vh3
            ex1_all2 = AllData[1,...] @ U4 @ Vh4
            ex2_all2 = AllData[2,...] @ U5 @ Vh5
            # obtain latent dynamics without averaging across trials
            alltar_latent_dynamics.append((id1,id2,id3,[[ex0_all1,ex1_all1], [ex0_all2,ex2_all1], [ex1_all2,ex2_all2]]))
            for tar in range(AllData.shape[1]):
                ex0 = np.mean(AllData[0, tar, ...], axis=0)
                ex1 = np.mean(AllData[1, tar, ...], axis=0)
                ex2 = np.mean(AllData[2, tar, ...], axis=0)
                ex0_1 = ex0 @ U0 @ Vh0
                ex1_1 = ex1 @ U1 @ Vh1
                ex0_2 = ex0 @ U2 @ Vh2
                ex2_1 = ex2 @ U3 @ Vh3
                ex1_2 = ex1 @ U4 @ Vh4
                ex2_2 = ex2 @ U5 @ Vh5
                latent_dynamics.append((id1, id2, id3, tar, [[ex0_1,ex1_1], [ex0_2,ex2_1], [ex1_2,ex2_2]]))
            print('pair ' + str(num) + ' done')
            num += 1
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_latent_dynamics_aligned_0.03_30_drift_2subj.npy',latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_alltar_cue_latent_dynamics_aligned_0.03_30_drift_2subj.npy',alltar_latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_CCA_score_aligned_0.03_30_drift_2subj.npy',CCA_score)

print('save successfully')