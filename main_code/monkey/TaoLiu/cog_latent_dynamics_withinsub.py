import mat73
import os, sys, pathlib
import numpy as np
from monkey.defs import *
import pyaldata as pyal
import warnings
from sklearn.decomposition import PCA
from cca_zoo.models import GCCA
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
GCCA_score = []
AllVAF = []
warnings.filterwarnings("ignore")
num = 1
# for id1,id2,testList in pairIndex_uni:
for subj in pairIndex_uni:
    model = PCA(n_components=n_components, svd_solver='full')
    data_list = [allDFs[subj[0]]] + [allDFs[subj[1]]] + [allDFs[subj[2]]]
    AllData = np.empty((len(data_list), target_num, n_shared_trial, n_timepoints, model.n_components))
    VAF = np.empty((len(subj),model.n_components))
    for session_num, data_ in enumerate(data_list):
        rates = np.concatenate(data_['Spikes_count'], axis=0)
        rates_model = model.fit(rates)
        VAF[session_num,:] = np.cumsum(rates_model.explained_variance_ratio_) # obtaining VAF
        data_['_pca'] = [model.transform(s) for s in data_['Spikes_count']]

        target_num = np.size(np.unique(data_['Labels']))
        for targetIdx in range(target_num):
            all_id = data_['trial_id'][data_['Labels'] == targetIdx + 1]
            while ((all_id_sh0 := rng.permutation(all_id)) == all_id).all():
                continue
            all_id = all_id_sh0
            data_tmp = [np.array(data_['_pca'])[data_['trial_id'] == trial_id] for trial_id in all_id[:n_shared_trial]]
            for trial, trial_rates in enumerate(data_tmp):
                AllData[session_num, targetIdx, trial, :, :] = np.squeeze(trial_rates)
    AllVAF.append(VAF)
    data1 = np.reshape(AllData[0, ...], (-1, defs.n_components))
    data2 = np.reshape(AllData[1, ...], (-1, defs.n_components))
    data3 = np.reshape(AllData[2, ...], (-1, defs.n_components))
    gcca = GCCA(latent_dims=defs.n_components)
    gcca.fit([data1, data2, data3]) # align across 3 sessions
    GCCA_score.append((subj[0], subj[1], subj[2], [gcca.score([data1, data2, data3])])) # computing GCCA scores
    U0, s0, Vh0 = svd(gcca.weights[0], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
    U1, s1, Vh1 = svd(gcca.weights[1], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
    U2, s2, Vh2 = svd(gcca.weights[2], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
    ex0_all = AllData[0, ...] @ U0 @ Vh0 # project to latent space
    ex1_all = AllData[1, ...] @ U1 @ Vh1
    ex2_all = AllData[2, ...] @ U2 @ Vh2
    alltar_latent_dynamics.append((subj[0], subj[1], subj[2], [ex0_all, ex1_all, ex2_all]))
    for tar in range(AllData.shape[1]):
        ex0 = np.mean(AllData[0,tar, ...], axis=0) # mean latent dynamics across trials
        ex1 = np.mean(AllData[1,tar, ...], axis=0)
        ex2 = np.mean(AllData[2,tar, ...], axis=0)
        ex0 = ex0 @ U0 @ Vh0
        ex1 = ex1 @ U1 @ Vh1
        ex2 = ex2 @ U2 @ Vh2
        latent_dynamics.append((subj[0], subj[1], subj[2], tar, [ex0, ex1, ex2]))
    print('pair '+str(num)+' done')
    num += 1
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_latent_dynamics_withinsub_0.03_30_drift.npy',latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_alltar_cue_latent_dynamics_withinsub_0.03_30_drift.npy',alltar_latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_GCCA_score_withinsub_0.03_30_drift.npy',GCCA_score)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_VAF_0.03_30_drift.npy',AllVAF)

print('save successfully')