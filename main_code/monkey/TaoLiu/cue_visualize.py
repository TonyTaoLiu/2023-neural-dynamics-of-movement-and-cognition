import os, sys, pathlib
from pprint import pprint
import gc
import pickle
from importlib import reload
import logging, warnings
import params
logging.basicConfig(level=logging.ERROR)

import pandas as pd
import numpy as np
import sklearn
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import scipy.linalg as linalg

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from tools.utilityTools import *

def multimat_correlation(mat):
    '''
    this function is for calculating the correlation between multiple matrices. the idea is to expand the matrix first, then
    compute the correlation between the expanded vectors. The vectors can be concated to a new matrix as the input of numpy.correlation
    input: mat, list of matrices
    output: corr_mat, matrix containing correlation coefficients between input matrices
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


def find_animal_indices(pairIndex,animalIndex):
    '''
    This function is to divide the pair index into several groups according to different monkeys used to train classifier
    Parameters
    ----------
    pairIndex, list of pair index
    animalIndex, list, information of which monkey the sessions belong to

    Returns
    -------
    animal_indices, list of sessions which has been divided into different groups according to monkeys
    '''
    pairIndex_expand = []
    for pair in pairIndex:
        for jj in range(len(pair[2])):
            pairIndex_expand.append([pair[0], pair[1], pair[2][jj]])
    pairIndex_expand = np.array(pairIndex_expand)

    animal_indices = [[],[],[]]
    for i in range(np.size(pairIndex_expand,0)):
        if pairIndex_expand[i,0] in animalIndex[0]: animal_indices[0].append(i)
        elif pairIndex_expand[i,0] in animalIndex[1]: animal_indices[1].append(i)
        elif pairIndex_expand[i,0] in animalIndex[2]: animal_indices[2].append(i)

    return animal_indices

#######################################################################################################################
# cue
#######################################################################################################################
# visualization of aligned latent dynamics in 3D space
aligned_latent_dynamics = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_latent_dynamics_aligned_0.03_30_drift.npy',allow_pickle=True)
colors = get_colors(8,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(4):
    id1, id2, id3, tar, aligned_example = aligned_latent_dynamics[i, :]
    for j, ex in enumerate(aligned_example):
        axes[j].plot(ex[:, 0], ex[:, 1], ex[:, 2], color=colors[tar], lw=1) # using only the first 3 principle components

titles = [r'Monkey B (aligned)', r'Monkey K (aligned)', r'Monkey L (aligned)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(f'CC1', labelpad=-10)
    ax.set_ylabel(f'CC2', labelpad=-10)
    ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')
# fig1.show()
# fig2.show()
# fig3.show()
fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyB_cognitive-cue-aligned-0.03-drift-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyK_cognitive-cue-aligned-0.03-drift-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyL_cognitive-cue-aligned-0.03-drift-example.eps',
    format='eps', dpi=1000)
fig1.clf()
fig2.clf()
fig3.clf()
plt.close()

# visualization of unaligned latent dynamics in 3D space
unaligned_latent_dynamics = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_latent_dynamics_unaligned_0.03_30_drift.npy',allow_pickle=True)
colors = get_colors(8,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(4):
    id1, id2, id3, tar, unaligned_example = unaligned_latent_dynamics[i, :]
    for j, ex in enumerate(unaligned_example):
        axes[j].plot(ex[:, 0], ex[:, 1], ex[:, 2], color=colors[tar], lw=1) # using only the first 3 principle components

titles = [r'Monkey B (unaligned)', r'Monkey K (unaligned)', r'Monkey L (unaligned)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(f'CC1', labelpad=-10)
    ax.set_ylabel(f'CC2', labelpad=-10)
    ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')
# fig1.show()
# fig2.show()
# fig3.show()
fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyB_cognitive-cue-unaligned-0.03-drift-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyK_cognitive-cue-unaligned-0.03-drift-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyL_cognitive-cue-unaligned-0.03-drift-example.eps',
    format='eps', dpi=1000)
fig1.clf()
fig2.clf()
fig3.clf()
plt.close()



# visualization of aligned latent dynamics in 3D space (CCA)
aligned_latent_dynamics = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_latent_dynamics_aligned_0.03_30_drift_2subj.npy',allow_pickle=True)
colors = get_colors(8,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
fig4 = plt.figure(4)
ax4 = plt.axes(projection='3d', fc='None')
fig5 = plt.figure(5)
ax5 = plt.axes(projection='3d', fc='None')
fig6 = plt.figure(6)
ax6 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for i in range(4):
    id1, id2, id3, tar, aligned_example = aligned_latent_dynamics[i, :]
    for j, ex in enumerate(aligned_example):
        axes[2*j].plot(ex[0][:, 0], ex[0][:, 1], ex[0][:, 2], color=colors[tar], lw=1) # using only the first 3 principle components
        axes[2*j+1].plot(ex[1][:, 0], ex[1][:, 1], ex[1][:, 2], color=colors[tar], lw=1)

titles = [r'Monkey B (B x K)', r'Monkey K (B x K)',
          r'Monkey B (B x L)', r'Monkey L (B x L)',
          r'Monkey K (K x L)', r'Monkey L (K x L)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(f'CC1', labelpad=-10)
    ax.set_ylabel(f'CC2', labelpad=-10)
    ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')
# fig1.show()
# fig2.show()
# fig3.show()
fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyB_cognitive-cue-aligned-0.03-drift-BK-example.eps',format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyK_cognitive-cue-aligned-0.03-drift-BK-example.eps',format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyB_cognitive-cue-aligned-0.03-drift-BL-example.eps',format='eps', dpi=1000)
fig4.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyL_cognitive-cue-aligned-0.03-drift-BL-example.eps',format='eps', dpi=1000)
fig5.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyK_cognitive-cue-aligned-0.03-drift-KL-example.eps',format='eps', dpi=1000)
fig6.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyL_cognitive-cue-aligned-0.03-drift-KL-example.eps',format='eps', dpi=1000)
fig1.clf()
fig2.clf()
fig3.clf()
fig4.clf()
fig5.clf()
fig6.clf()
plt.close()


# results of cognitive cue dataset
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
          'cognitive_prefrontal8A_cue_cls_scores_set_0.03_30_drift.pickle', 'rb') as f:
    cls_scores_aligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
          'cognitive_prefrontal8A_cue_cls_scores_set_unaligned_0.03_30_drift.pickle', 'rb') as f:
    cls_scores_unaligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
          'cognitive_prefrontal8A_cue_cls_scores_set_within_0.03_30_drift.pickle', 'rb') as f:
    cls_scores_within = pickle.load(f)

avg_cls_scores = [[],[],[]]
std_cls_scores = [[],[],[]]
cls_scores = [[],[],[]]
for pairNum in range(len(cls_scores_aligned)):
    cls_scores[0].append(np.mean(np.array(cls_scores_unaligned[pairNum][3]),1))
    cls_scores[1].append(np.mean(np.array(cls_scores_aligned[pairNum][3]),1))

pairIndex = []
for pairNum in range(len(cls_scores_within)):
    pairIndex.append(cls_scores_within[pairNum][0])
    cls_scores[2].append(np.mean(np.array(cls_scores_within[pairNum][3]),1))

avg_cls_scores[0].append(np.mean(np.array(cls_scores[0]),0))
std_cls_scores[0].append(np.std(np.array(cls_scores[0]),0))
avg_cls_scores[1].append(np.mean(np.array(cls_scores[1]),0))
std_cls_scores[1].append(np.std(np.array(cls_scores[1]),0))
temp = np.array(cls_scores[2])
avg_cls_scores[2].append(np.mean(temp,1))
std_cls_scores[2].append(np.std(temp,1))

# visualization
y_cls = np.squeeze(np.array(avg_cls_scores))
y_cls_std = np.squeeze(np.array(std_cls_scores))
x = np.arange(3)+1
x_label = ['unaligned','aligned','within']
width = 0.3
fig, ax = plt.subplots(figsize=(7,12))
rects1 = ax.bar(x - width, y_cls[:,0], width, label='Classifier B')
ax.errorbar(x - width,y_cls[:,0],yerr=y_cls_std[:,0],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects2 = ax.bar(x, y_cls[:,1], width, label='Classifier K')
ax.errorbar(x,y_cls[:,1],yerr=y_cls_std[:,1],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects3 = ax.bar(x + width, y_cls[:,2], width, label='Classifier L')
ax.errorbar(x + width,y_cls[:,2],yerr=y_cls_std[:,2],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_title('GCCA cue',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=15)
plt.ylim([0,1])
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-cls-cue-sep-classifier-0.03.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


#########################################################################################################################
# 2subj
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
          'cognitive_prefrontal8A_cue_cls_scores_set_0.03_30_drift_2subj.pickle', 'rb') as f:
    aligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
          'cognitive_prefrontal8A_cue_cls_scores_set_unaligned_0.03_30_drift.pickle', 'rb') as f:
    unaligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
          'cognitive_prefrontal8A_cue_cls_scores_set_within_0.03_30_drift_2subj.pickle', 'rb') as f:
    withinsub_cls_scores = pickle.load(f)

avg_cls_scores = [[],[],[]]
std_cls_scores = [[],[],[]]
cls_scores = [[],[],[]]
for pairNum in range(len(aligned_cls_scores)):
    cls_scores[0].append(np.mean(np.array(unaligned_cls_scores[pairNum][3]),1))
    cls_scores[1].append(np.mean(np.array(aligned_cls_scores[pairNum][3]),1))

pairIndex = []
for pairNum in range(len(withinsub_cls_scores)):
    pairIndex.append(withinsub_cls_scores[pairNum][0])
    cls_scores[2].append(np.mean(np.array(withinsub_cls_scores[pairNum][3]),1))

avg_cls_scores[0].append(np.mean(np.array(cls_scores[0]),0))
std_cls_scores[0].append(np.std(np.array(cls_scores[0]),0))
avg_cls_scores[1].append(np.mean(np.array(cls_scores[1]),0))
std_cls_scores[1].append(np.std(np.array(cls_scores[1]),0))
temp = np.array(cls_scores[2])
avg_cls_scores[2].append(np.mean(temp,1))
std_cls_scores[2].append(np.std(temp,1))

# visualization
# classification
y_cls = np.squeeze(np.array(avg_cls_scores))
y_cls_std = np.squeeze(np.array(std_cls_scores))
x = np.arange(3)+1
x_label = ['unaligned','aligned','within']
width = 0.3
fig, ax = plt.subplots(figsize=(7,12))
rects1 = ax.bar(x - width, y_cls[:,0], width, label='Classifier B')
ax.errorbar(x - width,y_cls[:,0],yerr=y_cls_std[:,0],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects2 = ax.bar(x, y_cls[:,1], width, label='Classifier K')
ax.errorbar(x,y_cls[:,1],yerr=y_cls_std[:,1],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects3 = ax.bar(x + width, y_cls[:,2], width, label='Classifier L')
ax.errorbar(x + width,y_cls[:,2],yerr=y_cls_std[:,2],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
plt.ylim([0,1])
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_title('CCA cue',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(loc='upper left',fontsize=15)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-cls-cue-sep-classifier-0.03-2subj.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# GCCA score
from tools import utilityTools as utility
aligned_GCCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_GCCA_score_aligned_0.03_30_drift.npy',allow_pickle=True)
unaligned_GCCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_corr_score_unaligned_0.03_30_drift.npy',allow_pickle=True)
withinsub_GCCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_GCCA_score_withinsub_0.03_30_drift.npy',allow_pickle=True)
aligned_score = []
unaligned_score = []
for i in range(aligned_GCCA_score.shape[0]):
    _,_,_,aligned_temp = aligned_GCCA_score[i]
    _,_,_,unaligned_temp = unaligned_GCCA_score[i]
    aligned_score.append(aligned_temp[0])
    unaligned_score.append(sorted(unaligned_temp,reverse=True))

within_score = []
for i in range(withinsub_GCCA_score.shape[0]):
    _,_,_,within_temp = withinsub_GCCA_score[i]
    within_score.append(within_temp[0])

# fig,ax = plt.subplots(ncols=1,dpi=300)
# utility.shaded_errorbar(ax, np.arange(1,within_temp[0].shape[0]+1), np.array(within_score).T, color='gray',
#                         label='Within')
# utility.shaded_errorbar(ax, np.arange(1,aligned_temp[0].shape[0]+1), np.array(aligned_score).T, color='b',
#                         label='Aligned')
# utility.shaded_errorbar(ax, np.arange(1,unaligned_temp.shape[0]+1), np.array(unaligned_score).T, color='cornflowerblue',
#                         label='Unaligned')

fig,ax = plt.subplots(ncols=1,dpi=300)
utility.shaded_errorbar(ax, np.arange(1,11), np.array(within_score)[:,:10].T, color='gray',
                        label='Within')
utility.shaded_errorbar(ax, np.arange(1,11), np.array(aligned_score)[:,:10].T, color='b',
                        label='Aligned')
utility.shaded_errorbar(ax, np.arange(1,11), np.array(unaligned_score)[:,:10].T, color='cornflowerblue',
                        label='Unaligned')

ax.set_ylim([0,1])
# ax.set_xlim([.5,unaligned_temp.shape[0]+.5])
ax.set_xlim([.5,10+.5])
ax.set_xlabel('Component order',fontdict={'size':15})
ax.set_title('Cue GCCA score of components',fontdict={'size':15})
ax.legend(fontsize=10)
ax.set_ylabel('Correlation',fontdict={'size':15})
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-cue-component-score-drift-0.03-10.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


########################################################################################################################
# CCA score
from tools import utilityTools as utility
aligned_CCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_CCA_score_aligned_0.03_30_drift_2subj.npy',allow_pickle=True)
unaligned_CCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_corr_score_unaligned_0.03_30_drift.npy',allow_pickle=True)
withinsub_CCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
    'cog_prefrontal8A_cue_CCA_score_withinsub_0.03_30_drift_2subj.npy',allow_pickle=True)
aligned_score = []
unaligned_score = []
for i in range(aligned_CCA_score.shape[0]):
    _,_,_,aligned_temp = aligned_CCA_score[i]
    _,_,_,unaligned_temp = unaligned_CCA_score[i]
    aligned_score.append(np.mean(np.array(aligned_temp),0))
    unaligned_score.append(sorted(unaligned_temp,reverse=True))

within_score = []
for i in range(withinsub_CCA_score.shape[0]):
    _,_,_,within_temp = withinsub_CCA_score[i]
    within_score.append(np.mean(np.array(within_temp),0))

# fig,ax = plt.subplots(ncols=1,dpi=300)
# utility.shaded_errorbar(ax, np.arange(1,within_temp[0].shape[0]+1), np.array(within_score).T, color='gray',
#                         label='Within')
# utility.shaded_errorbar(ax, np.arange(1,aligned_temp[0].shape[0]+1), np.array(aligned_score).T, color='b',
#                         label='Aligned')
# utility.shaded_errorbar(ax, np.arange(1,unaligned_temp.shape[0]+1), np.array(unaligned_score).T, color='cornflowerblue',
#                         label='Unaligned')

fig,ax = plt.subplots(ncols=1,dpi=300)
utility.shaded_errorbar(ax, np.arange(1,11), np.array(within_score)[:,:10].T, color='gray',
                        label='Within')
utility.shaded_errorbar(ax, np.arange(1,11), np.array(aligned_score)[:,:10].T, color='b',
                        label='Aligned')
utility.shaded_errorbar(ax, np.arange(1,11), np.array(unaligned_score)[:,:10].T, color='cornflowerblue',
                        label='Unaligned')

ax.set_ylim([0,1])
# ax.set_xlim([.5,unaligned_temp.shape[0]+.5])
ax.set_xlim([.5,10+.5])
ax.set_xlabel('Component order',fontdict={'size':15})
ax.set_title('Cue CCA score of components',fontdict={'size':15})
ax.legend(fontsize=10)
ax.set_ylabel('Correlation',fontdict={'size':15})
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-cue-component-score-drift-0.03-10-2subj.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# VAF
from tools import utilityTools as utility
VAF = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result\cognitive\cue/'
              'cog_prefrontal8A_cue_VAF_0.03_30_drift.npy',allow_pickle=True)

fig1,ax1 = plt.subplots(ncols=1,dpi=300)
fig2,ax2 = plt.subplots(ncols=1,dpi=300)
fig3,ax3 = plt.subplots(ncols=1,dpi=300)
axes = [ax1, ax2, ax3]
title = ['Monkey B','Monkey K','Monkey L']
col = ['cornflowerblue','tab:orange','palevioletred']
for i in range(VAF.shape[0]):
    x_ = np.arange(1, VAF.shape[-1] + 1)
    utility.shaded_errorbar(axes[i], x_, VAF[i].T, color=col[i], marker='o', zorder=1, label='PCA')

    axes[i].set_ylim([-.05, 1])
    axes[i].set_xlim([.6, VAF.shape[-1] + .6])
    axes[i].set_xlabel('Component')
    axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[i].set_ylabel('PCA VAF')
    axes[i].set_title(title[i])
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_bounds([1, VAF[0].shape[1]])
    axes[i].spines['left'].set_bounds([0, 1])

# fig1.show()
# fig2.show()
# fig3.show()

fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyB_cognitive-cue-PCA-VAF-0.03.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyK_cognitive-cue-PCA-VAF-0.03.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyL_cognitive-cue-PCA-VAF-0.03.eps',
    format='eps', dpi=1000)
fig1.clf()
fig2.clf()
fig3.clf()
plt.close()


# VAF-cc
from tools import utilityTools as utility
VAF_cc = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result\cognitive/cue/'
                 'cog_prefrontal8A_cue_VAF_cc_0.4_100_drift.npy',allow_pickle=True)

fig1,ax1 = plt.subplots(ncols=1,dpi=300)
fig2,ax2 = plt.subplots(ncols=1,dpi=300)
fig3,ax3 = plt.subplots(ncols=1,dpi=300)
axes = [ax1, ax2, ax3]
title = ['Monkey B','Monkey K','Monkey L']
col = ['cornflowerblue','tab:orange','palevioletred']
for i in range(VAF_cc.shape[1]):
    x_ = np.arange(1, VAF_cc.shape[-1] + 1)
    utility.shaded_errorbar(axes[i], x_, VAF_cc[:,i,:].T, color=col[i], marker='o', zorder=1, label='CCA')

    axes[i].set_ylim([-.05, 1])
    axes[i].set_xlim([.6, VAF_cc.shape[-1] + .6])
    axes[i].set_xlabel('Component')
    axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[i].set_ylabel('CCA VAF')
    axes[i].set_title(title[i])
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_bounds([1, VAF_cc[:,0,:].shape[1]])
    axes[i].spines['left'].set_bounds([0, 1])

# fig1.show()
# fig2.show()
# fig3.show()

fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyB_cognitive-cue-CCA-VAF.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyK_cognitive-cue-CCA-VAF.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyL_cognitive-cue-CCA-VAF.eps',
    format='eps', dpi=1000)
fig1.clf()
fig2.clf()
fig3.clf()
plt.close()


# average GCCA score
from tools import utilityTools as utility
GCCA_score_avg_cue = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/cue/'
                             'cog_prefrontal8A_cue_GCCA_score_avg_0.03_30_drift.npy')
GCCA_score_avg_target = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/target/'
                                'cog_prefrontal8A_target_GCCA_score_avg_0.03_30_drift.npy')
GCCA_score_avg_target_delay = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/cognitive/target/'
                                      'cog_prefrontal8A_target_GCCA_score_avg_0.03_30_drift_delay.npy')
fig,ax = plt.subplots(ncols=1,dpi=300)
utility.shaded_errorbar(ax, np.arange(5,5*GCCA_score_avg_target_delay.shape[0]+1,5), GCCA_score_avg_target_delay,
                        color='tab:red',label='Target-delay')
utility.shaded_errorbar(ax, np.arange(5,5*GCCA_score_avg_target.shape[0]+1,5), GCCA_score_avg_target,
                        color='tab:orange',label='Target')
utility.shaded_errorbar(ax, np.arange(5,5*GCCA_score_avg_cue.shape[0]+1,5), GCCA_score_avg_cue,
                        color='palevioletred',label='Cue')

ax.set_ylim([0,1])
ax.set_xlim([4.5,5*GCCA_score_avg_target.shape[0]+.5])
ax.set_xlabel('Manifold dimensionality',fontdict={'size':15})
ax.set_title('Preserved latent dynamics across individuals \nfor a range of neural manifold dimensionalities',
             fontdict={'size':15})
ax.legend(fontsize=10)
ax.set_ylabel('Mean across top four \ncanonical correlations',fontdict={'size':15})
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-GCCA-score-avg-0.03.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()