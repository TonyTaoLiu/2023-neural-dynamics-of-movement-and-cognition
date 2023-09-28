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
from scipy.linalg import svd, inv
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


def VAF_pc_cc(X: np.ndarray, C: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Calculate Variance Accounted For (VAF) for a double projection (as in from PCA --> to CCA) using the method in Gallego, NatComm, 2018

    Parameters
    ----------
    `X`: the data matrix, T x n with _T_ time points and _n_ neurons, and each neuron is **zero mean**.

    `C`: the first projection matrix, usually it is the `PCA_model.components_`, but in principle could be any projection matrix with orthogonal bases.

    `A` : is the CCA canonical axes, the output of the `canoncorr` function, in principle could be any projection matrix, not necessarily orthogonal.

    Returns
    -------
    `VAFs`: np.array with VAF for each axes of `C`, normalised between 0<VAF<1 for each axis, `sum(VAFs)` equals to total VAF.
    """
    # following the notation in Gallego 2018
    norm = lambda m: np.sum(m ** 2)

    VAFs = np.empty((C.shape[0],))
    for comp in range(1, C.shape[0] + 1):
        D = inv(A[:,:comp].T @ A[:,:comp]) @ A[:,:comp].T @ C
        E = C.T @ A[:,:comp]
        VAF = norm(X - X @ E @ D) / norm(X)
        VAFs[comp - 1] = 1 - VAF

    VAFs = np.array([VAFs[0], *np.diff(VAFs)])
    return VAFs


def VAF_gcca(data_list, n_components):
    '''
    Calculate the VAF of GCCA (from PCA to GCCA)
    Parameters
    ----------
    data_list, list of spike matrix of each session
    n_components, output dimension of GCCA

    Returns
    -------
    VAFs, list of VAF of each component
    VAFs_cumsum, the cumulative sum of VAFs
    '''
    rates = []
    rates_c = []
    pca_data = []
    for session_num, data_ in enumerate(data_list):
        rates.append(np.concatenate(data_['Spikes_count'],axis=0))
        rates_model = PCA(n_components=n_components,svd_solver='full').fit(rates[session_num]) # build PCA model
        rates_c.append(rates_model.components_)
        data_['_pca'] = [rates_model.transform(s) for s in data_['Spikes_count']] # perform PCA
        pca_data.append(np.concatenate(data_['_pca'],axis=0))

    n_samples = min([pca_data[i].shape[0] for i in range(len(pca_data))])
    pca_data = [pca_data[i][:n_samples,:] for i in range(len(pca_data))] # align the number of samples

    gcca = GCCA(latent_dims=defs.n_components)
    gcca.fit(pca_data) # align across subjects

    VAFs = []
    VAFs_cumsum = []
    for num in range(len(rates)):
        VAFs.append(VAF_pc_cc(rates[num], rates_c[num], gcca.weights[num])) # compute VAF of canonical components
        VAFs_cumsum.append(VAFs[num].cumsum()) # compute cumulative sum of VAFs

    return VAFs, VAFs_cumsum


data_path = '../../../cognitive data/actualUsed'
file_name = 'Extracted_spikes_target_drift_30.mat' # change file name for analysis of different periods

# change the start and end point of window when analyzing different periods
# allDFs0,full_list0,n_shared_trial0,target_num0,n_timepoints0 = load_data(data_path,file_name,1,12)
allDFs1,full_list1,n_shared_trial1,target_num1,n_timepoints1 = load_data(data_path,file_name,84,100)
# allDFs2,full_list2,n_shared_trial2,target_num2,n_timepoints2 = load_data(data_path,file_name,23,34)

pairIndex_uni = [[],[],[]]
for i, (animal, session) in enumerate(full_list1):
    if 'B' in animal: pairIndex_uni[0].append(i)
    if 'K' in animal: pairIndex_uni[1].append(i)
    if 'L' in animal: pairIndex_uni[2].append(i)

rng = params.rng
VAF_cc = []
warnings.filterwarnings("ignore")
num = 1
for id1 in pairIndex_uni[0]:
    for id2 in pairIndex_uni[1]:
        for id3 in pairIndex_uni[2]:
            data_list = [allDFs1[id1]] + [allDFs1[id2]] + [allDFs1[id3]]
            VAFs, VAFs_cumsum = VAF_gcca(data_list,n_components) # calculate VAFs and their cumulative sum
            VAF_cc.append(VAFs_cumsum)
            print('pair ' + str(num) + ' done')
            num += 1
# change with your own saving path
np.save('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
        'cog_prefrontal8A_target_VAF_cc_0.03_30_drift_delay.npy',VAF_cc)
