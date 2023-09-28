import numpy as np
import pickle
import matplotlib.pyplot as plt
from tools.utilityTools import *
import math

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


def correlation_matrix(arr,subjects,title=None,cmap=plt.cm.Blues):
    '''
    This function is to plot array as heatmap. For this project, the input array should be a squared matrix, with upper
    triangle matrix being the aligned correlations, the bottom triangle matrix being the unaligned correlations, row and
    collum are both subjects
    Parameters
    ----------
    arr: squared matrix, upper triangle part contains aligned correlations, lower triangle part contains unaligned
    correlations. The diagonal elements are all 1.
    subjects: list containing names of all subjects
    title: title of the graph, default None
    cmap: colormap used for plotting graph, default plt.cm.Blues
    Returns
    -------
    plot showing the input array
    '''
    cm = arr
    fig,ax = plt.subplots(figsize=(5,4),dpi=300)
    im = ax.imshow(cm,interpolation='nearest',cmap=cmap)
    cb = fig.colorbar(im,ax=ax)
    cb.set_label('Correlation',fontsize=15)
    cb.ax.tick_params(labelsize=12)
    # ax.set(xticks=np.arange(cm.shape[1]),
    #        yticks=np.arange(cm.shape[0]),
    #        # ... and label them with the respective list entries
    #        xticklabels=subjects, yticklabels=subjects,
    #        title=title,
    #        ylabel='Subjects',
    #        xlabel='Subjects')
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(subjects,fontsize=12)
    ax.set_yticklabels(subjects,fontsize=12)
    ax.set_xlabel('Subjects',fontsize=15)
    ax.set_ylabel('Subjects',fontsize=15)
    ax.set_title(title,fontsize=15)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    fig.tight_layout()

    return ax



# visualization of aligned latent dynamics in 3D space
# change file name for analyzing different periods
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_latent_dynamics_target_aligned_pca30_full_reshape_3d_delay2.pickle', 'rb') as f:
    latent_dynamics_aligned = pickle.load(f)
colors = get_colors(2,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(2):
    id1, id2, id3, tar, aligned_example = latent_dynamics_aligned[i]
    for j, ex in enumerate(aligned_example):
        for trial in range(ex.shape[0]):
            # axes[j].plot(ex[trial,:,0], ex[trial,:,1], ex[trial,:,2], color=colors[tar], lw=1)
            axes[j].scatter(ex[trial,:,0], ex[trial,:,1], ex[trial,:,2], color=colors[tar], s=0.5,alpha=0.75) # using only the first 3 principle components

titles = [r'Monkey B (aligned)', r'Monkey K (aligned)', r'Monkey L (aligned)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_xlabel(f'CC1', labelpad=-10)
    # ax.set_ylabel(f'CC2', labelpad=-10)
    # ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')
# fig1.show()
# fig2.show()
# fig3.show()
fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyB_cognitive-target-aligned-example-delay.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyK_cognitive-target-aligned-example-delay.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyL_cognitive-target-aligned-example-delay.eps',
    format='eps', dpi=1000)
fig1.clf()
fig2.clf()
fig3.clf()
plt.close()


# visualization of unaligned latent dynamics in 3D space
# change file name for analyzing different periods
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
          'cognitive/target/cog_latent_dynamics_target_unaligned_pca30_full_reshape_3d2.pickle', 'rb') as f:
    latent_dynamics_unaligned = pickle.load(f)

colors = get_colors(2,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(2):
    id1, id2, id3, tar, unaligned_example = latent_dynamics_unaligned[i]
    for j, ex in enumerate(unaligned_example):
        for trial in range(ex.shape[0]):
            # axes[j].plot(ex[trial,:,0], ex[trial,:,1], ex[trial,:,2], color=colors[tar], lw=1)
            axes[j].scatter(ex[trial,:,0], ex[trial,:,1], ex[trial,:,2], color=colors[tar], s=0.5,alpha=0.75) # using only the first 3 principle components

titles = [r'Monkey B (unaligned)', r'Monkey K (unaligned)', r'Monkey L (unaligned)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_xlabel(f'CC1', labelpad=-10)
    # ax.set_ylabel(f'CC2', labelpad=-10)
    # ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')

fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyB_target-unaligned-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyK_target-unaligned-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyL_target-unaligned-example.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# visualization of within latent dynamics in 3D space
# change file name for analyzing different periods
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
          'cognitive/target/cog_latent_dynamics_target_within_pca30_full_reshape_3d_delay.pickle', 'rb') as f:
    latent_dynamics_within = pickle.load(f)

colors = get_colors(2,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(6):
    _, tar, within_example = latent_dynamics_within[i]
    for j, ex in enumerate(within_example):
        for trial in range(ex.shape[0]):
            # axes[j].plot(ex[trial,:,0], ex[trial,:,1], ex[trial,:,2], color=colors[tar], lw=1)
            axes[math.floor(i/2)].scatter(ex[trial, :, 0], ex[trial, :, 1], ex[trial, :, 2], color=colors[tar], s=0.5,
                            alpha=0.75)  # using only the first 3 principle components

titles = [r'Monkey B (within)', r'Monkey K (within)', r'Monkey L (within)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_xlabel(f'CC1', labelpad=-10)
    # ax.set_ylabel(f'CC2', labelpad=-10)
    # ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')

fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyB_target-within-example-delay.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyK_target-within-example-delay.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyL_target-within-example-delay.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()

# classification results
# change file name for analyzing different periods
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_aligned_CEBRA.pickle', 'rb') as f:
    cls_scores_aligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_unaligned_CEBRA.pickle', 'rb') as f:
    cls_scores_unaligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_within_CEBRA.pickle', 'rb') as f:
    cls_scores_within = pickle.load(f)

avg_cls_scores = [[],[],[]]
std_cls_scores = [[],[],[]]
cls_scores = [[],[],[]]
for pairNum in range(len(cls_scores_aligned)):
    # averaged accuracy of each classifier (train on 1 monkey and test on 2 monkeys)
    cls_scores[0].append(np.mean(np.array(cls_scores_unaligned[pairNum][3]),1))
    cls_scores[1].append(np.mean(np.array(cls_scores_aligned[pairNum][3]),1))

pairIndex = []
for pairNum in range(len(cls_scores_within)):
    # averaged accuracy of each classifier (train on 1 session and test on 2 sessions)
    pairIndex.append(cls_scores_within[pairNum][0])
    cls_scores[2].append(np.mean(np.array(cls_scores_within[pairNum][3]),1))

avg_cls_scores[0].append(np.mean(np.array(cls_scores[0]),0)) # averaged results across groups
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
ax.set_title('Average classification performance\nunder different conditions (CEBRA)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=15)
plt.ylim([0,1])
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/cog-classification-target-sep-classifier-CEBRA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# comparsion CCA
# change file name for analyzing different periods
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_aligned_CEBRA_comparsion_CCA.pickle', 'rb') as f:
    aligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_unaligned_CEBRA_comparsion_GCCA.pickle', 'rb') as f:
    unaligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_within_CEBRA_comparsion_CCA.pickle', 'rb') as f:
    withinsub_cls_scores = pickle.load(f)

avg_cls_scores = [[],[],[]]
std_cls_scores = [[],[],[]]
cls_scores = [[],[],[]]
for pairNum in range(len(aligned_cls_scores)):
    # averaged accuracy of each classifier (train on 1 monkey and test on 2 monkeys)
    cls_scores[0].append(np.mean(np.array(unaligned_cls_scores[pairNum][3]),1))
    cls_scores[1].append(np.mean(np.array(aligned_cls_scores[pairNum][3]),1))

pairIndex = []
for pairNum in range(len(withinsub_cls_scores)):
    # averaged accuracy of each classifier (train on 1 session and test on 2 sessions)
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
ax.set_title('Average classification performance\nunder different conditions (CCA)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(loc='upper left',fontsize=15)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-classification-target-sep-classifier-CEBRA-comparsion-CCA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# comparsion GCCA
# change file name for analyzing different periods
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_aligned_CEBRA_comparsion_GCCA.pickle', 'rb') as f:
    aligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_unaligned_CEBRA_comparsion_GCCA.pickle', 'rb') as f:
    unaligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/target/'
          'cog_Prefrontal8A_cls_scores_target_pca30_within_CEBRA_comparsion_GCCA.pickle', 'rb') as f:
    withinsub_cls_scores = pickle.load(f)

avg_cls_scores = [[],[],[]]
std_cls_scores = [[],[],[]]
cls_scores = [[],[],[]]
for pairNum in range(len(aligned_cls_scores)):
    # averaged accuracy of each classifier (train on 1 monkey and test on 2 monkeys)
    cls_scores[0].append(np.mean(np.array(unaligned_cls_scores[pairNum][3]),1))
    cls_scores[1].append(np.mean(np.array(aligned_cls_scores[pairNum][3]),1))

pairIndex = []
for pairNum in range(len(withinsub_cls_scores)):
    # averaged accuracy of each classifier (train on 1 session and test on 2 sessions)
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
ax.set_title('Average classification performance\nunder different conditions (GCCA)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(loc='upper left',fontsize=15)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-classification-target-sep-classifier-CEBRA-comparsion-GCCA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()