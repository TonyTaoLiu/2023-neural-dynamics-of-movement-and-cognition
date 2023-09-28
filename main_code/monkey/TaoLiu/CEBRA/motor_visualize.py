import pickle
import numpy as np
import matplotlib.pyplot as plt
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


def find_animal_indices_noexpand(pairIndex_expand,animalIndex):
    '''
    This function is to divide the pair index into several groups according to different monkeys used to train classifier
    Parameters ()
    ----------
    pairIndex_expend, list of pair index
    animalIndex, list, information of which monkey the sessions belong to

    Returns
    -------
    animal_indices_4, list of sessions which has been divided into different groups according to monkeys, the recordings
    of left and right hemisphere of Chewie are seemed as different monkeys
    animal_indices_3, list of sessions which has been divided into different groups according to monkeys, the recordings
    of left and right hemisphere of Chewie are seemed as same monkeys
    '''
    animal_indices_4 = [[],[],[],[]]
    animal_indices_3 = [[],[],[]]
    for i in range(len(pairIndex_expand)):
        if pairIndex_expand[i][0] in animalIndex[0]:
            animal_indices_4[0].append(i)
            animal_indices_3[0].append(i)
        elif pairIndex_expand[i][0] in animalIndex[1]:
            animal_indices_4[1].append(i)
            animal_indices_3[0].append(i)
        elif pairIndex_expand[i][0] in animalIndex[2]:
            animal_indices_4[2].append(i)
            animal_indices_3[1].append(i)
        elif pairIndex_expand[i][0] in animalIndex[3]:
            animal_indices_4[3].append(i)
            animal_indices_3[2].append(i)

    return animal_indices_4, animal_indices_3


def find_animal_indices(pairIndex,animalIndex):
    '''
    This function is to divide the pair index into several groups according to different monkeys used to train classifier
    Parameters ()
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

# combining all pairs into one group
# this code is used when separating one run of alignment with CEBRA in several times
with open('E:\MSc Project\monkey dataset\motor/motor_CEBRA_data_aligned_pca10.pickle', 'rb') as f:
    data = pickle.load(f)
latent_dynamics_full = []
latent_dynamics_avg = []
for group_num in range(72):
    id1, id2, id3, AllData = data[group_num]
    with open(
            'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
            'motor_latent_dynamics_aligned_pca10_full'+str(group_num)+'.pickle','rb') as f:
        latent_dynamics_full.append(pickle.load(f))

    multi_embeddings = latent_dynamics_full[group_num][3]
    for i in range(8):
        target_embeddings = []
        for j in range(len(multi_embeddings)):
            trial_avg = multi_embeddings[j].reshape(8,AllData[0].shape[1],AllData[0].shape[2],10)[i,...].mean(axis=0)
            # trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
            target_embeddings.append(trial_avg)
        latent_dynamics_avg.append((id1,id2,id3,i,target_embeddings))

with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor/motor_latent_dynamics_aligned_pca10_full.pickle','wb') as f:
    pickle.dump(latent_dynamics_full, f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor/motor_latent_dynamics_aligned_pca10_avg.pickle','wb') as f:
    pickle.dump(latent_dynamics_avg, f)


########################################################################################################################
# classification results
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
          'motor_MCx_CEBRA_cls_scores_pca10.pickle', 'rb') as f:
    cls_scores_aligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
          'motor_MCx_CEBRA_cls_scores_pca10_unaligned.pickle', 'rb') as f:
    cls_scores_unaligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
          'motor_MCx_CEBRA_cls_scores_pca10_within.pickle', 'rb') as f:
    cls_scores_within = pickle.load(f)

with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
          'motor_CEBRA_animalIndex_within.pickle', 'rb') as f:
    animal_index = pickle.load(f)

avg_cls_scores = [[],[],[]]
std_cls_scores = [[],[],[]]
cls_scores = [[],[],[]]
for pairNum in range(len(cls_scores_aligned)):
    cls_scores[0].append(np.mean(np.array(cls_scores_unaligned[pairNum][3]),1)) # averaged results of classifiers trained on each monkeys
    cls_scores[1].append(np.mean(np.array(cls_scores_aligned[pairNum][3]),1))

pairIndex = []
for pairNum in range(len(cls_scores_within)):
    pairIndex.append(cls_scores_within[pairNum][0])
    cls_scores[2].append(np.mean(np.array(cls_scores_within[pairNum][1]),1))

animal_indices_4, animal_indices_3 = find_animal_indices_noexpand(pairIndex,animal_index) # divide groups according to training monkey
avg_cls_scores[0].append(np.mean(np.array(cls_scores[0]),0)) # averaged results across groups
std_cls_scores[0].append(np.std(np.array(cls_scores[0]),0))
avg_cls_scores[1].append(np.mean(np.array(cls_scores[1]),0))
std_cls_scores[1].append(np.std(np.array(cls_scores[1]),0))
temp = np.array(cls_scores[2])
avg_cls_scores[2].append([np.mean(temp[animal_indices_3[k],:]) for k in range(len(animal_indices_3))]) # not separating Chewie
std_cls_scores[2].append([np.std(temp[animal_indices_3[k],:]) for k in range(len(animal_indices_3))])

# visualization
y_cls = np.squeeze(np.array(avg_cls_scores))
y_cls_std = np.squeeze(np.array(std_cls_scores))
x = np.arange(3)+1
x_label = ['unaligned','aligned','within']
width = 0.3
fig, ax = plt.subplots(figsize=(7,12))
rects1 = ax.bar(x - width, y_cls[:,0], width, label='Classifier C')
ax.errorbar(x - width,y_cls[:,0],yerr=y_cls_std[:,0],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects2 = ax.bar(x, y_cls[:,1], width, label='Classifier M')
ax.errorbar(x,y_cls[:,1],yerr=y_cls_std[:,1],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects3 = ax.bar(x + width, y_cls[:,2], width, label='Classifier J')
ax.errorbar(x + width,y_cls[:,2],yerr=y_cls_std[:,2],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_title('Average classification performance \n under different conditions (CEBRA)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=15)
plt.ylim([0,1.05])
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/motor-classification-acc-sep-classifier-CEBRA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# comparsion CCA
aligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor\comparsion/'
    'motor_M1_cls_scores_set_CEBRA_compare_CCA.npy', allow_pickle=True)
unaligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor\comparsion/'
    'motor_M1_cls_scores_set_unaligned_CEBRA_compare_GCCA.npy', allow_pickle=True)
withinsub_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor\comparsion/'
    'motor_M1_cls_scores_set_withinsub_CEBRA_compare_CCA.npy', allow_pickle=True)

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
fig, ax = plt.subplots(figsize=(7,12))
rects1 = ax.bar(x - width, y_cls[:,0], width, label='Classifier C')
ax.errorbar(x - width,y_cls[:,0],yerr=y_cls_std[:,0],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects2 = ax.bar(x, y_cls[:,1], width, label='Classifier M')
ax.errorbar(x,y_cls[:,1],yerr=y_cls_std[:,1],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects3 = ax.bar(x + width, y_cls[:,2], width, label='Classifier J')
ax.errorbar(x + width,y_cls[:,2],yerr=y_cls_std[:,2],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
plt.ylim([0,1.05])
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_title('Average classification performance under different conditions',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(loc='upper left',fontsize=15)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/motor-classification-acc-sep-classifier-CEBRA-comparsion-CCA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# comparsion GCCA
aligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor\comparsion/'
    'motor_M1_cls_scores_set_CEBRA_compare_GCCA.npy', allow_pickle=True)
unaligned_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor\comparsion/'
    'motor_M1_cls_scores_set_unaligned_CEBRA_compare_GCCA.npy', allow_pickle=True)
withinsub_cls_scores = np.load(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor\comparsion/'
    'motor_M1_cls_scores_set_withinsub_CEBRA_compare_GCCA.npy', allow_pickle=True)

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
fig, ax = plt.subplots(figsize=(7,12))
rects1 = ax.bar(x - width, y_cls[:,0], width, label='Classifier C')
ax.errorbar(x - width,y_cls[:,0],yerr=y_cls_std[:,0],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects2 = ax.bar(x, y_cls[:,1], width, label='Classifier M')
ax.errorbar(x,y_cls[:,1],yerr=y_cls_std[:,1],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
rects3 = ax.bar(x + width, y_cls[:,2], width, label='Classifier J')
ax.errorbar(x + width,y_cls[:,2],yerr=y_cls_std[:,2],ecolor='red',fmt='o',elinewidth=1.5,capsize=5)
plt.ylim([0,1.05])
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_title('Average classification performance under different conditions',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(loc='upper left',fontsize=15)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/motor-classification-acc-sep-classifier-CEBRA-comparsion-GCCA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()

########################################################################################################################
# visualization of aligned latent dynamics in 3D space
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_latent_dynamics_aligned_pca10_full_3d_reshape.pickle', 'rb') as f:
    latent_dynamics_aligned = pickle.load(f)

colors = get_colors(8)
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]

id1, id2, id3, aligned_example = latent_dynamics_aligned[0]
for j, ex in enumerate(aligned_example):
    for trial in range(ex.shape[1]):
        for tar in range(8):
            axes[j].scatter(ex[tar,trial,:,0], ex[tar,trial,:,1], ex[tar,trial,:,2], color=colors[tar], s=0.5,
                            alpha=0.75)  # using only the first 3 principle components
            # axes[j].plot(ex[tar,trial,:,0], ex[tar,trial,:,1], ex[tar,trial,:,2], color=colors[tar], lw=1)

titles = [r'Monkey C (aligned)', r'Monkey M (aligned)', r'Monkey J (aligned)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_xlabel(f'CC1', labelpad=-10)
    # ax.set_ylabel(f'CC2', labelpad=-10)
    # ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')

fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyC_motor-aligned-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyM_motor-aligned-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyJ_motor-aligned-example.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# visualization of unaligned latent dynamics in 3D space
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_latent_dynamics_unaligned_pca10_reshape_3d.pickle', 'rb') as f:
    latent_dynamics_unaligned = pickle.load(f)

colors = get_colors(8)
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for subj in range(len(latent_dynamics_unaligned)):
    _, unaligned_example = latent_dynamics_unaligned[subj]
    for trial in range(unaligned_example[0].shape[1]):
        for tar in range(unaligned_example[0].shape[0]):
            axes[subj].scatter(unaligned_example[0][tar,trial,:,0],unaligned_example[0][tar,trial,:,1],
                            unaligned_example[0][tar,trial,:,2], color=colors[tar], s=0.5,alpha=0.75)

titles = [r'Monkey C (unaligned)', r'Monkey M (unaligned)', r'Monkey J (unaligned)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_xlabel(f'CC1', labelpad=-10)
    # ax.set_ylabel(f'CC2', labelpad=-10)
    # ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')

fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyC_motor-unaligned-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyM_motor-unaligned-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyJ_motor-unaligned-example.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# visualization of within
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_latent_dynamics_within_pca10_reshape_3d.pickle', 'rb') as f:
    latent_dynamics_within = pickle.load(f)

colors = get_colors(8)
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]

# repeat 3 times, each time refers to one monkey
_, within_example = latent_dynamics_within[0]
for sess in range(len(within_example)):
    for trial in range(within_example[0].shape[1]):
        for tar in range(within_example[0].shape[0]):
            axes[0].scatter(within_example[sess][tar,trial,:,0],within_example[sess][tar,trial,:,1],
                            within_example[sess][tar,trial,:,2], color=colors[tar], s=0.5,alpha=0.75)

_, within_example = latent_dynamics_within[8]
for sess in range(len(within_example)):
    for trial in range(within_example[0].shape[1]):
        for tar in range(within_example[0].shape[0]):
            axes[1].scatter(within_example[sess][tar, trial, :, 0], within_example[sess][tar, trial, :, 1],
                            within_example[sess][tar, trial, :, 2], color=colors[tar], s=0.5, alpha=0.75)

_, within_example = latent_dynamics_within[9]
for sess in range(len(within_example)):
    for trial in range(within_example[0].shape[1]):
        for tar in range(within_example[0].shape[0]):
            axes[2].scatter(within_example[sess][tar, trial, :, 0], within_example[sess][tar, trial, :, 1],
                            within_example[sess][tar, trial, :, 2], color=colors[tar], s=0.5, alpha=0.75)

titles = [r'Monkey C (within)', r'Monkey M (within)', r'Monkey J (within)']
for i, ax in enumerate(axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_xlabel(f'CC1', labelpad=-10)
    # ax.set_ylabel(f'CC2', labelpad=-10)
    # ax.set_zlabel(f'CC3', labelpad=-10)
    ax.set_title(titles[i], pad=0, loc='center')

fig1.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyC_motor-within-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyM_motor-within-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyJ_motor-within-example.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()