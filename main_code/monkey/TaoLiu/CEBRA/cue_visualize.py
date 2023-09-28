import numpy as np
import pickle
import matplotlib.pyplot as plt
from tools.utilityTools import *
import math

# classification results
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_aligned_CEBRA.pickle', 'rb') as f:
    cls_scores_aligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_unaligned_CEBRA.pickle', 'rb') as f:
    cls_scores_unaligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_within_CEBRA.pickle', 'rb') as f:
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
ax.set_title('Average classification performance\nunder different conditions (CEBRA)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=15)
plt.ylim([0,1.05])
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/cog-classification-cue-sep-classifier-CEBRA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# comparsion CCA
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_aligned_CEBRA_comparsion_CCA.pickle', 'rb') as f:
    aligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_unaligned_CEBRA_comparsion_GCCA.pickle', 'rb') as f:
    unaligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_within_CEBRA_comparsion_CCA.pickle', 'rb') as f:
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
plt.ylim([0,1.05])
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_title('Average classification performance\nunder different conditions (CCA)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(loc='upper left',fontsize=15)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-classification-cue-sep-classifier-CEBRA-comparsion-CCA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# comparsion GCCA
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_aligned_CEBRA_comparsion_GCCA.pickle', 'rb') as f:
    aligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_unaligned_CEBRA_comparsion_GCCA.pickle', 'rb') as f:
    unaligned_cls_scores = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_Prefrontal8A_cls_scores_cue_pca30_within_CEBRA_comparsion_GCCA.pickle', 'rb') as f:
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
plt.ylim([0,1.05])
ax.set_ylabel('Accuracy',fontsize=20)
ax.set_title('Average classification performance\nunder different conditions (GCCA)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(x_label,rotation=30,fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(loc='upper left',fontsize=15)
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/cog-classification-cue-sep-classifier-CEBRA-comparsion-GCCA.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# visualization of aligned latent dynamics in 3D space
# perhaps needs to change to CEBRA visualization
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/cognitive/cue/'
          'cog_latent_dynamics_cue_aligned_pca30_full_reshape_3d2.pickle', 'rb') as f:
    latent_dynamics_aligned = pickle.load(f)
colors = get_colors(8,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(4):
    id1, id2, id3, tar, aligned_example = latent_dynamics_aligned[i]
    for j, ex in enumerate(aligned_example):
        for trial in range(ex.shape[0]):
            # axes[j].plot(ex[:, 0], ex[:, 1], ex[:, 2], color=colors[tar], lw=1)
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
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyB_cognitive-cue-aligned-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyK_cognitive-cue-aligned-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyL_cognitive-cue-aligned-example.eps',
    format='eps', dpi=1000)
fig1.clf()
fig2.clf()
fig3.clf()
plt.close()


# visualization of unaligned latent dynamics in 3D space
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
          'cognitive/cue/cog_latent_dynamics_cue_unaligned_pca30_full_reshape_3d2.pickle', 'rb') as f:
    latent_dynamics_unaligned = pickle.load(f)

colors = get_colors(8,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(4):
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
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyB_cue-unaligned-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyK_cue-unaligned-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyL_cue-unaligned-example.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()


# visualization of within latent dynamics in 3D space
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
          'cognitive/cue/cog_latent_dynamics_cue_within_pca30_full_reshape_3d.pickle', 'rb') as f:
    latent_dynamics_within = pickle.load(f)

colors = get_colors(8,colormap='tab20')
fig1 = plt.figure(1)
ax1 = plt.axes(projection='3d', fc='None')
fig2 = plt.figure(2)
ax2 = plt.axes(projection='3d', fc='None')
fig3 = plt.figure(3)
ax3 = plt.axes(projection='3d', fc='None')
axes = [ax1, ax2, ax3]
for i in range(12):
    _, tar, within_example = latent_dynamics_within[i]
    for j, ex in enumerate(within_example):
        for trial in range(ex.shape[0]):
            # axes[j].plot(ex[trial,:,0], ex[trial,:,1], ex[trial,:,2], color=colors[tar], lw=1)
            axes[math.floor(i/4)].scatter(ex[trial, :, 0], ex[trial, :, 1], ex[trial, :, 2], color=colors[tar], s=0.5,
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
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyB_cue-within-example.eps',
    format='eps', dpi=1000)
fig2.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyK_cue-within-example.eps',
    format='eps', dpi=1000)
fig3.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/figure/monkeyL_cue-within-example.eps',
    format='eps', dpi=1000)
plt.clf()
plt.close()