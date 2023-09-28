import pickle
import numpy as np
import matplotlib.pyplot as plt
from tools.utilityTools import *

# visualization of canonical correlations and classification accuracies changed with session numbers

# GCCA score
from tools import utilityTools as utility
GCCA_scores = []
for i in range(2,7):
    with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result\cognitive\effect_num_sess/target/'
              'cognitive_prefrontal8A_target_canonical_scores_set_'+str(i)+'_0.03_30_drift.pickle', 'rb') as f:
        GCCA_scores.append(pickle.load(f))

names = ['Monkey B','Monkey K','Monkey L']
figs = []
for subj in range(3):
    fig, ax = plt.subplots(ncols=1, dpi=300)
    for i in range(2,7):
        # utility.shaded_errorbar(ax,np.arange(1,11),np.mean(np.array(GCCA_scores[i-2]),1)[:,:10].T,label=str(i))
        ax.plot(np.arange(1, 11),np.mean(np.array(GCCA_scores[i-2]),1)[subj,:10],label=str(i))
    ax.set_ylim([0, 0.9])
    # ax.set_xlim([4.5,5*GCCA_score_avg_target.shape[0]+.5])
    ax.set_xlabel('Component order', fontdict={'size': 15})
    ax.set_title(names[subj], fontdict={'size': 15})
    ax.legend(fontsize=10)
    ax.set_ylabel('Canonical Correlations', fontdict={'size': 15})
    fig.tight_layout()
    figs.append(fig)
    plt.savefig(
        'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'+
        names[subj]+'_cog-target-GCCA-score-2-6-0.03.eps',format='eps', dpi=1000)
    plt.clf()
    plt.close()


# classification performance when aligning across different number of sessions
cls_scores = []
for i in range(2,7):
    with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result\cognitive\effect_num_sess/target-delay/'
              'cognitive_prefrontal8A_target_cls_scores_set_'+str(i)+'_0.03_30_drift_delay.pickle', 'rb') as f:
        cls_scores.append(np.array(pickle.load(f)))

names = ['Monkey B', 'Monkey K', 'Monkey L']
scores_avg = []
scores_std = []
for i in range(2,7):
    # temp = np.mean(np.mean(cls_scores[i-2],-1),-1)
    temp = cls_scores[i-2].reshape((cls_scores[i-2].shape[0],-1))
    scores_avg.append(np.mean(temp,1))
    scores_std.append(np.std(temp,1))

fig, ax = plt.subplots(ncols=1, dpi=300)
for subj in range(3):
    ax.errorbar(np.arange(2,7),np.array(scores_avg)[:,subj].T,yerr=np.array(scores_std)[:,subj].T,
                fmt='o-',elinewidth=2,capsize=4,label=names[subj])
ax.set_ylim([0.45,1])
ax.set_xticks(np.arange(2,7))
ax.set_xlabel('Number of sessions', fontdict={'size': 15})
ax.set_title('Effect of number of sessions on classification', fontdict={'size': 15})
ax.legend(loc='lower right',fontsize=10)
ax.set_ylabel('Accuracy', fontdict={'size': 15})
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'cog-target-cls-score-2-6-0.03-delay.eps',format='eps', dpi=1000)
plt.clf()
plt.close()


# motor GCCA scores
GCCA_scores = []
for i in range(2,5):
    with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result\motor/'
              'motor_MCx_canonical_scores_set_'+str(i)+'.pickle', 'rb') as f:
        GCCA_scores.append(pickle.load(f))

names = ['Monkey C1','Monkey C2']
figs = []
for subj in range(2):
    fig, ax = plt.subplots(ncols=1, dpi=300)
    for i in range(2,5):
        # utility.shaded_errorbar(ax,np.arange(1,11),np.mean(np.array(GCCA_scores[i-2]),1)[:,:10].T,label=str(i))
        ax.plot(np.arange(1, 11),np.mean(np.array(GCCA_scores[i-2]),1)[subj,:10],label=str(i))
    ax.set_ylim([0, 1])
    # ax.set_xlim([4.5,5*GCCA_score_avg_target.shape[0]+.5])
    ax.set_xlabel('Component order', fontdict={'size': 15})
    ax.set_title(names[subj], fontdict={'size': 15})
    ax.legend(fontsize=10)
    ax.set_ylabel('Canonical Correlations', fontdict={'size': 15})
    fig.tight_layout()
    figs.append(fig)
    plt.savefig(
        'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'+
        names[subj]+'_motor-MCx-GCCA-score-2-4.eps',format='eps', dpi=1000)
    plt.clf()
    plt.close()


# motor classification accuracies when aligning across different number of sessions
cls_scores = []
for i in range(2,5):
    with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result\motor/'
              'motor_MCx_cls_scores_set_'+str(i)+'.pickle', 'rb') as f:
        cls_scores.append(np.array(pickle.load(f)))

names = ['Monkey C1', 'Monkey C2']
scores_avg = []
scores_std = []
for i in range(2,5):
    # temp = np.mean(np.mean(cls_scores[i-2],-1),-1)
    temp = cls_scores[i-2].reshape((cls_scores[i-2].shape[0],-1))
    scores_avg.append(np.mean(temp,1))
    scores_std.append(np.std(temp,1))

fig, ax = plt.subplots(ncols=1, dpi=300)
for subj in range(2):
    ax.errorbar(np.arange(2,5),np.array(scores_avg)[:,subj].T,yerr=np.array(scores_std)[:,subj].T,
                fmt='o-',elinewidth=2,capsize=4,label=names[subj])

ax.set_xticks(np.arange(2,5))
ax.set_xlabel('Number of sessions', fontdict={'size': 15})
ax.set_title('Effect on classification', fontdict={'size': 15})
ax.legend(loc='lower right',fontsize=10)
ax.set_yticks(np.arange(0.6,1.1,0.1))
ax.set_ylabel('Accuracy', fontdict={'size': 15})
ax.set_ylim([0.6,1.01])
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'motor-MCx-cls-score-2-4.eps',format='eps', dpi=1000)
plt.clf()
plt.close()