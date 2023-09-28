import numpy as np
import pyaldata as pyal
import mat73
import os, sys, pathlib
from sklearn.decomposition import PCA
import params

def load_data(data_path, file_name, start, end):
    allDFs = []
    full_list = []
    n_shared_trial = np.inf
    dirs = [d for d in os.listdir(data_path) if
            os.path.isdir(os.path.join(data_path, d))]  # obtain all directories names
    for full_name in dirs:
        animal = full_name[0]
        session = full_name[1:]
        data = mat73.loadmat(data_path + '/' + full_name + '/' + file_name) # load data
        data = data['Extracted_spikes']
        data['animal'] = animal # add animal name
        data['session'] = session # add session name
        data['trial_id'] = np.arange(1, data['Spikes_count'].shape[0] + 1) # add trial id
        data['Spikes_count'] = np.transpose(data['Spikes_count'], (0, 2, 1))
        data['Spikes_count'] = data['Spikes_count'][:, :,
                               np.mean(np.abs(np.concatenate(data['Spikes_count'], 0)) / 0.03, 0) > 1] # remove low firing neurons
        data['Spikes_count'] = np.array(
            [np.sqrt(data['Spikes_count'][ii, ...]) for ii in range(data['Spikes_count'].shape[0])]) # square root transformation
        # data['Spikes_count'][np.isnan(data['Spikes_count'])] = 0
        win = pyal.norm_gauss_window(0.03, 0.05)
        data['Spikes_count'] = np.array(
            [pyal.smooth_data(data['Spikes_count'][ii, ...], win=win, backend='convolve1d') / 0.03 for ii in
             range(data['Spikes_count'].shape[0])]) # computing firing rate in smooth manner
        # del data['Spikes']
        data['Spikes_count'] = data['Spikes_count'][:, start:end, :]

        allDFs.append(data)
        full_list.append((animal, session))

        target_num = np.size(np.unique(data['Labels']))
        for target in range(1, target_num):
            data_tmp = data['Spikes_count'][data['Labels'] == target]
            n_shared_trial = min(data_tmp.shape[0], n_shared_trial)

    n_shared_trial = int(n_shared_trial) # find out shared number of trials across sessions
    n_timepoints = data['Spikes_count'].shape[1]

    return allDFs, full_list, n_shared_trial, target_num, n_timepoints


def get_data_mat(data_list, target_num, n_shared_trial, n_timepoints, n_components):
    rng = params.rng
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
