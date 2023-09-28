import mat73
import os, sys, pathlib
import numpy as np
from monkey.defs import *
import pyaldata as pyal
import warnings
from scipy.stats import zscore
from sklearn.decomposition import PCA
from cca_zoo.models import GCCA
from sklearn.manifold import TSNE
from scipy.linalg import svd


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


def get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, n_components):
    model = PCA(n_components=n_components, svd_solver='full')
    AllData = np.empty((len(data_list), target_num, n_shared_trial, n_timepoints, model.n_components))
    for session_num, data_ in enumerate(data_list):
        rates = np.concatenate(data_['Spikes_count'], axis=0)
        rates_model = model.fit(rates)
        data_['_pca'] = [model.transform(s) for s in data_['Spikes_count']] # perform pca

        target_num = np.size(np.unique(data_['Labels']))
        for targetIdx in range(target_num):
            all_id = data_['trial_id'][data_['Labels'] == targetIdx + 1]
            while ((all_id_sh0 := rng.permutation(all_id)) == all_id).all():
                continue
            all_id = all_id_sh0
            data_tmp = [np.array(data_['_pca'])[data_['trial_id'] == trial_id] for trial_id in all_id[:n_shared_trial]]
            for trial, trial_rates in enumerate(data_tmp):
                AllData[session_num, targetIdx, trial, :, :] = np.squeeze(trial_rates)

    return AllData

allDFs = []
full_list = []
n_shared_trial = np.inf
data_path = '../../../cognitive data/actualUsed'
dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))] # obtain all directories names
for full_name in dirs:
    animal = full_name[0]
    session = full_name[1:] # add session name
    data = mat73.loadmat(data_path+'/'+full_name+'/Extracted_spikes_cue_drift_30.mat') # load data
    data = data['Extracted_spikes']
    data['animal'] = animal # add animal name
    data['session'] = session # add session name
    data['trial_id'] = np.arange(1,data['Spikes_count'].shape[0]+1) # add trial id
    data['Spikes_count'] = np.transpose(data['Spikes_count'], (0, 2, 1))
    # remove low firing neurons
    data['Spikes_count'] = data['Spikes_count'][:,:,np.mean(np.abs(np.concatenate(data['Spikes_count'], 0)) / 0.03, 0) > 1]
    # square root transformation
    data['Spikes_count'] = np.array([np.sqrt(data['Spikes_count'][ii,...]) for ii in range(data['Spikes_count'].shape[0])])
    win = pyal.norm_gauss_window(0.03, 0.05)
    data['Spikes_count'] = np.array(
        [pyal.smooth_data(data['Spikes_count'][ii, ...], win=win, backend='convolve1d') / 0.03 for ii in
         range(data['Spikes_count'].shape[0])]) # computing firing rate in smooth manner
    data['Spikes_count'] = data['Spikes_count'][:, 50:66, :] # select windows for analysis

    allDFs.append(data)
    full_list.append((animal,session))

    target_num = np.size(np.unique(data['Labels']))
    for target in range(1,target_num):
        data_tmp = data['Spikes_count'][data['Labels'] == target]
        n_shared_trial = min(data_tmp.shape[0],n_shared_trial)

n_shared_trial = int(n_shared_trial) # find out shared number of trials across sessions
n_timepoints = data['Spikes_count'].shape[1]

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
warnings.filterwarnings("ignore")
num = 1
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            data_list = [allDFs[id1]] + [allDFs[id2]] + [allDFs[id3]]
            AllData = get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, 30) # perform pca
            data1 = np.reshape(AllData[0, ...], (-1, defs.n_components))
            data2 = np.reshape(AllData[1, ...], (-1, defs.n_components))
            data3 = np.reshape(AllData[2, ...], (-1, defs.n_components))
            gcca = GCCA(latent_dims=defs.n_components)
            gcca.fit([data1, data2, data3]) # align across 3 subjects
            GCCA_score.append((id1, id2, id3, [gcca.score([data1, data2, data3])]))
            U0, s0, Vh0 = svd(gcca.weights[0], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U1, s1, Vh1 = svd(gcca.weights[1], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            U2, s2, Vh2 = svd(gcca.weights[2], full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
            ex0_all = AllData[0, ...] @ U0 @ Vh0 # project to latent space
            ex1_all = AllData[1, ...] @ U1 @ Vh1
            ex2_all = AllData[2, ...] @ U2 @ Vh2
            alltar_latent_dynamics.append((id1, id2, id3, [ex0_all, ex1_all, ex2_all])) # obtain latent dynamics without averaging across trials
            for tar in range(AllData.shape[1]):
                ex0 = np.mean(AllData[0, tar, ...], axis=0) # mean latent dynamics across trials
                ex1 = np.mean(AllData[1, tar, ...], axis=0)
                ex2 = np.mean(AllData[2, tar, ...], axis=0)
                ex0 = ex0 @ U0 @ Vh0
                ex1 = ex1 @ U1 @ Vh1
                ex2 = ex2 @ U2 @ Vh2
                latent_dynamics.append((id1, id2, id3, tar, [ex0, ex1, ex2]))
                # ex0_2d = tsne.fit_transform(ex0)
                # ex1_2d = tsne.fit_transform(ex1)
                # ex2_2d = tsne.fit_transform(ex2)
                # tsne_latent_dynamics.append((id1, id2, id3, tar, [ex0_2d, ex1_2d, ex2_2d]))
            print('pair ' + str(num) + ' done')
            num += 1

# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_latent_dynamics_aligned_0.03_30_drift.npy',latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_alltar_cue_latent_dynamics_aligned_0.03_30_drift.npy',alltar_latent_dynamics)
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cog_prefrontal8A_cue_GCCA_score_aligned_0.03_30_drift.npy',GCCA_score)

print('save successfully')