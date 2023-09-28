
import os, sys, pathlib
from pprint import pprint
import gc
import pickle
from importlib import reload
import logging, warnings

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
from scipy.stats import pearsonr


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


#########################################################################################################################
# visualization of aligned latent dynamics in 3D space
aligned_latent_dynamics = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/motor_M1_aligned_latent_dynamics.npy',
    allow_pickle=True)
colors = get_colors(8)
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(8):
    id1, id2, id3, tar, aligned_example = aligned_latent_dynamics[i, :]
    for j, ex in enumerate(aligned_example):
        axes[j].plot(ex[:, 0], ex[:, 1], ex[:, 2], color=colors[tar], lw=1)

titles = [r'Monkey C (aligned)', r'Monkey M (aligned)', r'Monkey J (aligned)']
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
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyC_motor-aligned-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyM_motor-aligned-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyJ_motor-aligned-example.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()

# visualization of unaligned latent dynamics in 3D space
unaligned_latent_dynamics = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/motor_M1_unaligned_latent_dynamics.npy',
    allow_pickle=True)
colors = get_colors(8)
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(8):
    id1, id2, id3, tar, unaligned_example = unaligned_latent_dynamics[i, :]
    for j, ex in enumerate(unaligned_example):
        axes[j].plot(ex[:, 0], ex[:, 1], ex[:, 2], color=colors[tar], lw=1)

titles = [r'Monkey C (unaligned)', r'Monkey M (unaligned)', r'Monkey J (unaligned)']
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
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyC_motor-unaligned-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyM_motor-unaligned-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/monkeyJ_motor-unaligned-example.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# visualization of aligned latent dynamics in 3D space （CCA）
aligned_latent_dynamics = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/'
    'motor_M1_aligned_latent_dynamics_2subj.npy',allow_pickle=True)
colors = get_colors(8)
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
for i in range(8):
    id1, id2, id3, tar, aligned_example = aligned_latent_dynamics[i, :]
    for j, ex in enumerate(aligned_example):
        axes[2*j].plot(ex[0][:, 0], ex[0][:, 1], ex[0][:, 2], color=colors[tar], lw=1) # using only the first 3 principle components
        axes[2*j+1].plot(ex[1][:, 0], ex[1][:, 1], ex[1][:, 2], color=colors[tar], lw=1)

titles = [r'Monkey C (C x M)', r'Monkey M (C x M)',
          r'Monkey C (C x J)', r'Monkey J (C x J)',
          r'Monkey M (M x J)', r'Monkey J (M x J)']
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
    'monkeyC_motor-aligned-CM-example.eps',format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyM_motor-aligned-CM-example.eps',format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyC_motor-aligned-CJ-example.eps',format='eps', dpi=1000)
fig4.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyJ_motor-aligned-CJ-example.eps',format='eps', dpi=1000)
fig5.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyM_motor-aligned-MJ-example.eps',format='eps', dpi=1000)
fig6.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'monkeyJ_motor-aligned-MJ-example.eps',format='eps', dpi=1000)
fig1.clf()
fig2.clf()
fig3.clf()
fig4.clf()
fig5.clf()
fig6.clf()
plt.close()


########################################################################################################################
# GCCA score
from tools import utilityTools as utility
aligned_GCCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_MCx_aligned_GCCA_score.npy',
    allow_pickle=True)
unaligned_GCCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_MCx_unaligned_corr_score.npy',
    allow_pickle=True)
withinsub_GCCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_MCx_withinsub_GCCA_score.npy',
    allow_pickle=True)
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

fig,ax = plt.subplots(ncols=1,dpi=300)
utility.shaded_errorbar(ax, np.arange(1,within_temp[0].shape[0]+1), np.array(within_score).T, color='gray',
                        label='Within')
utility.shaded_errorbar(ax, np.arange(1,aligned_temp[0].shape[0]+1), np.array(aligned_score).T, color='b',
                        label='Aligned')
utility.shaded_errorbar(ax, np.arange(1,unaligned_temp.shape[0]+1), np.array(unaligned_score).T, color='cornflowerblue',
                        label='Unaligned')

ax.set_ylim([0,1])
ax.set_xlim([.5,unaligned_temp.shape[0]+.5])
ax.set_xlabel('Component order',fontdict={'size':15})
ax.set_title('GCCA score of components',fontdict={'size':15})
ax.legend(fontsize=10)
ax.set_ylabel('Correlation',fontdict={'size':15})
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/motor-component-score.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# CCA score
from tools import utilityTools as utility
aligned_CCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/'
    'motor_MCx_aligned_CCA_score_2subj.npy',allow_pickle=True)
unaligned_CCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/'
    'motor_MCx_unaligned_corr_score.npy',allow_pickle=True)
withinsub_CCA_score = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/'
    'motor_MCx_withinsub_CCA_score_2subj.npy',allow_pickle=True)
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

fig,ax = plt.subplots(ncols=1,dpi=300)
utility.shaded_errorbar(ax, np.arange(1,within_temp[0].shape[0]+1), np.array(within_score).T, color='gray',
                        label='Within')
utility.shaded_errorbar(ax, np.arange(1,aligned_temp[0].shape[0]+1), np.array(aligned_score).T, color='b',
                        label='Aligned')
utility.shaded_errorbar(ax, np.arange(1,unaligned_temp.shape[0]+1), np.array(unaligned_score).T, color='cornflowerblue',
                        label='Unaligned')

ax.set_ylim([0,1])
ax.set_xlim([.5,unaligned_temp.shape[0]+.5])
ax.set_xlabel('Component order',fontdict={'size':15})
ax.set_title('CCA score of components',fontdict={'size':15})
ax.legend(fontsize=10)
ax.set_ylabel('Correlation',fontdict={'size':15})
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/motor-component-score-2subj.eps.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()



########################################################################################################################
# visualization of classification accuracy with exec epoch
aligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_aligned_prep_nohist.npy',
    allow_pickle=True)
unaligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_unaligned_prep_nohist.npy',
    allow_pickle=True)
withinsub_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_within_prep_nohist.npy',
    allow_pickle=True)

pairIndex_uni = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_aligned_pairIndex.npy',
                        allow_pickle=True)

pairIndex_within = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_withinsub_pairIndex.npy',
                        allow_pickle=True)

animal_index = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_pairIndex.npy',
                        allow_pickle=True)

animal_indices = find_animal_indices(pairIndex_uni,animal_index) # divide groups according to training monkey

avg_cls_scores = []
std_cls_scores = []

avg_repeat_cls = [[],[]]
std_repeat_cls = [[],[]]
for i in range(np.size(unaligned_cls_scores,0)):
    scores_tmp_cls = [[], []]
    for j in range(np.size(unaligned_cls_scores, 1)):
        scores_tmp_cls[0].append([unaligned_cls_scores[i,j,3]]) # results of each group
        scores_tmp_cls[1].append([aligned_cls_scores[i,j,3]])

    temp = np.mean(np.squeeze(np.array(scores_tmp_cls[0])),1)
    avg_repeat_cls[0].append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))]) # averaged results of classifiers trained on each monkeys
    std_repeat_cls[0].append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    temp = np.mean(np.squeeze(np.array(scores_tmp_cls[1])), 1)
    avg_repeat_cls[1].append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    std_repeat_cls[1].append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])

# prepared for repeating tests of classification (only 1 repeat here)
# repeating test refers to repeat the same classification process on same dataset for serveral times to avoid random error
avg_cls_scores.append(np.mean(np.array(avg_repeat_cls[0]),0))
avg_cls_scores.append(np.mean(np.array(avg_repeat_cls[1]),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls[0]),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls[1]),0))

animal_indices = find_animal_indices(pairIndex_within,animal_index) # divide groups for within case

avg_repeat_cls = []
std_repeat_cls = []
for i in range(np.size(withinsub_cls_scores,0)):
    scores_tmp_cls = []
    for j in range(np.size(withinsub_cls_scores, 1)):
        scores_tmp_cls.append([withinsub_cls_scores[i, j, 3]])

    temp = np.mean(np.squeeze(np.array(scores_tmp_cls)), 1)
    avg_repeat_cls.append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    std_repeat_cls.append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])

avg_cls_scores.append(np.mean(np.array(avg_repeat_cls),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls),0))

# visualization
# classification
y_cls = np.squeeze(np.array(avg_cls_scores))
y_cls_std = np.squeeze(np.array(std_cls_scores))
x = np.arange(3)+1
x_label = ['unaligned','aligned','within']
width = 0.3
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, y_cls[:,0], width, label='Classifier C')
ax.errorbar(x - width,y_cls[:,0],yerr=y_cls_std[:,0],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects2 = ax.bar(x, y_cls[:,1], width, label='Classifier M')
ax.errorbar(x,y_cls[:,1],yerr=y_cls_std[:,1],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects3 = ax.bar(x + width, y_cls[:,2], width, label='Classifier J')
ax.errorbar(x + width,y_cls[:,2],yerr=y_cls_std[:,2],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
plt.ylim([0,1.12])
ax.set_ylabel('Accuracy',fontsize=15)
ax.set_title('Average classification performance under different conditions',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(x_label,fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(loc='upper left',fontsize=10)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/motor-classification-acc-sep-classifier-prep.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# visualization of classification accuracy with exec epoch across 2 subjects
aligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_prep_nohist_2subj.npy',
    allow_pickle=True)
unaligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_unaligned_prep_nohist.npy',
    allow_pickle=True)
withinsub_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_withinsub_prep_nohist_2subj.npy',
    allow_pickle=True)

pairIndex_uni = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_aligned_pairIndex.npy',
                        allow_pickle=True)

pairIndex_within = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_withinsub_pairIndex.npy',
                        allow_pickle=True)

animal_index = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_pairIndex.npy',
                        allow_pickle=True)

animal_indices = find_animal_indices(pairIndex_uni,animal_index)

avg_cls_scores = []
std_cls_scores = []

avg_repeat_cls = [[],[]]
std_repeat_cls = [[],[]]
for i in range(np.size(unaligned_cls_scores,0)):
    scores_tmp_cls = [[], []]
    for j in range(np.size(unaligned_cls_scores, 1)):
        scores_tmp_cls[0].append([unaligned_cls_scores[i,j,3]])
        scores_tmp_cls[1].append([aligned_cls_scores[i,j,3]])

    temp = np.mean(np.squeeze(np.array(scores_tmp_cls[0])),1)
    avg_repeat_cls[0].append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    std_repeat_cls[0].append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    temp = np.mean(np.squeeze(np.array(scores_tmp_cls[1])), 1)
    avg_repeat_cls[1].append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    std_repeat_cls[1].append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])

avg_cls_scores.append(np.mean(np.array(avg_repeat_cls[0]),0))
avg_cls_scores.append(np.mean(np.array(avg_repeat_cls[1]),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls[0]),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls[1]),0))

animal_indices = find_animal_indices(pairIndex_within,animal_index)

avg_repeat_cls = []
std_repeat_cls = []
for i in range(np.size(withinsub_cls_scores,0)):
    scores_tmp_cls = []
    for j in range(np.size(withinsub_cls_scores, 1)):
        scores_tmp_cls.append([withinsub_cls_scores[i, j, 3]])

    temp = np.mean(np.squeeze(np.array(scores_tmp_cls)), 1)
    avg_repeat_cls.append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    std_repeat_cls.append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])

avg_cls_scores.append(np.mean(np.array(avg_repeat_cls),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls),0))

# visualization
# classification
y_cls = np.squeeze(np.array(avg_cls_scores))
y_cls_std = np.squeeze(np.array(std_cls_scores))
x = np.arange(3)+1
x_label = ['unaligned','aligned','within']
width = 0.3
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, y_cls[:,0], width, label='Classifier C')
ax.errorbar(x - width,y_cls[:,0],yerr=y_cls_std[:,0],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects2 = ax.bar(x, y_cls[:,1], width, label='Classifier M')
ax.errorbar(x,y_cls[:,1],yerr=y_cls_std[:,1],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects3 = ax.bar(x + width, y_cls[:,2], width, label='Classifier J')
ax.errorbar(x + width,y_cls[:,2],yerr=y_cls_std[:,2],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
plt.ylim([0,1.12])
ax.set_ylabel('Accuracy',fontsize=15)
ax.set_title('Average classification performance under different conditions',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(x_label,fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(loc='upper left',fontsize=10)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/motor-classification-acc-sep-classifier-exec-2subj.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()

# visualization of classification accuracy with exec epoch and lstm decoder
aligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_aligned_exec_2subj.npy',
    allow_pickle=True)
unaligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_unaligned_exec.npy',
    allow_pickle=True)
withinsub_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor/addition/motor_M1_cls_scores_set_withinsub_exec_2subj.npy',
    allow_pickle=True)

pairIndex_uni = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_aligned_pairIndex.npy',
                        allow_pickle=True)

pairIndex_within = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_M1_withinsub_pairIndex.npy',
                        allow_pickle=True)

animal_index = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/motor_pairIndex.npy',
                        allow_pickle=True)

animal_indices = find_animal_indices(pairIndex_uni,animal_index)

avg_cls_scores = []
std_cls_scores = []

avg_repeat_cls = [[],[]]
std_repeat_cls = [[],[]]
for i in range(np.size(unaligned_cls_scores,0)):
    scores_tmp_cls = [[], []]
    for j in range(np.size(unaligned_cls_scores, 1)):
        scores_tmp_cls[0].append([unaligned_cls_scores[i,j,3]])
        scores_tmp_cls[1].append([aligned_cls_scores[i,j,3]])

    temp = np.mean(np.squeeze(np.array(scores_tmp_cls[0])),1)
    avg_repeat_cls[0].append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    std_repeat_cls[0].append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    temp = np.mean(np.squeeze(np.array(scores_tmp_cls[1])), 1)
    avg_repeat_cls[1].append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    std_repeat_cls[1].append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])

avg_cls_scores.append(np.mean(np.array(avg_repeat_cls[0]),0))
avg_cls_scores.append(np.mean(np.array(avg_repeat_cls[1]),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls[0]),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls[1]),0))

animal_indices = find_animal_indices(pairIndex_within,animal_index)

avg_repeat_cls = []
std_repeat_cls = []
for i in range(np.size(withinsub_cls_scores,0)):
    scores_tmp_cls = []
    for j in range(np.size(withinsub_cls_scores, 1)):
        scores_tmp_cls.append([withinsub_cls_scores[i, j, 3]])

    temp = np.mean(np.squeeze(np.array(scores_tmp_cls)), 1)
    avg_repeat_cls.append([np.mean(temp[animal_indices[k]]) for k in range(len(animal_indices))])
    std_repeat_cls.append([np.std(temp[animal_indices[k]]) for k in range(len(animal_indices))])

avg_cls_scores.append(np.mean(np.array(avg_repeat_cls),0))
std_cls_scores.append(np.mean(np.array(std_repeat_cls),0))

# visualization
# classification
y_cls = np.squeeze(np.array(avg_cls_scores))
y_cls_std = np.squeeze(np.array(std_cls_scores))
x = np.arange(3)+1
x_label = ['unaligned','aligned','within']
width = 0.3
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, y_cls[:,0], width, label='Classifier C')
ax.errorbar(x - width,y_cls[:,0],yerr=y_cls_std[:,0],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects2 = ax.bar(x, y_cls[:,1], width, label='Classifier M')
ax.errorbar(x,y_cls[:,1],yerr=y_cls_std[:,1],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects3 = ax.bar(x + width, y_cls[:,2], width, label='Classifier J')
ax.errorbar(x + width,y_cls[:,2],yerr=y_cls_std[:,2],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
plt.ylim([0,1.12])
ax.set_ylabel('Accuracy',fontsize=15)
ax.set_title('Average classification performance under different conditions',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(x_label,fontsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(loc='upper left',fontsize=10)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/motor-classification-acc-sep-classifier-exec-lstm.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()



# average GCCA score
from tools import utilityTools as utility
GCCA_score_avg = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
                         'motor_MCx_aligned_GCCA_score_avg.npy')
fig,ax = plt.subplots(ncols=1,dpi=300)
utility.shaded_errorbar(ax, np.arange(5,5*GCCA_score_avg.shape[0]+1,5), GCCA_score_avg, color='cornflowerblue',
                        label='Motor')

ax.set_ylim([0,1])
ax.set_xlim([4.5,5*GCCA_score_avg.shape[0]+.5])
ax.set_xlabel('Manifold dimensionality',fontdict={'size':15})
ax.set_title('Preserved latent dynamics across individuals \nfor a range of neural manifold dimensionalities',
             fontdict={'size':15})
# ax.legend(fontsize=10)
ax.set_ylabel('Mean across top four \ncanonical correlations',fontdict={'size':15})
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/motor-GCCA-score-avg.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


from tools import utilityTools as utility
VAF = np.load('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/'
              'motor_MCx_VAF.npy',allow_pickle=True)
fig1,ax1 = plt.subplots(ncols=1,dpi=300)
fig2,ax2 = plt.subplots(ncols=1,dpi=300)
fig3,ax3 = plt.subplots(ncols=1,dpi=300)
axes = [ax1, ax2, ax3]
title = ['Monkey C','Monkey M','Monkey J']
col = ['cornflowerblue','tab:orange','palevioletred']
for i in range(VAF.shape[1]):
    x_ = np.arange(1, VAF.shape[-1] + 1)
    utility.shaded_errorbar(axes[i], x_, VAF[:,i,:].T, color=col[i], marker='o', zorder=1, label='PCA')

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