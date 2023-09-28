import numpy as np
import matplotlib.pyplot as plt

'''
Plot the classification results all on one graph for each condition (center-out, cue period, delay period, target period)
'''

# results of classification tests
motor_acc = [[0.155966667,0.863076667,0.91802],             # CCA-preparing
             [0.155966667,0.684,0.8335],                    # GCCA-preparing
             [0.087566667,0.9299,0.962533333],              # CCA-executing
             [0.087566667,0.752266667,0.8687]]              # GCCA-executing

target_acc = [[0.501566667,0.7816,0.806],                   # CCA-delay
              [0.501566667,0.761233333,0.7736],             # GCCA-delay
              [0.483166667,0.756166667,0.768233333],        # CCA-target
              [0.483166667,0.766866667,0.780466667]]        # GCCA-target

cue_acc = [[0.256,0.8109,0.801466667],                      # CCA-cue
           [0.256,0.816,0.821966667]]                       # GCCA-cue

motor_acc_std = [[0.062766667,0.042106667,0.01932],         # CCA-preparing
                 [0.062766667,0.073933333,0.048633333],     # GCCA-preparing
                 [0.044633333,0.022066667,0.0087],          # CCA-executing
                 [0.044633333,0.070133333,0.044533333]]     # GCCA-executing

target_acc_std = [[0.045733333,0.046933333,0.025466667],    # CCA-delay
                  [0.045733333,0.038733333,0.016466667],    # GCCA-delay
                  [0.043066667,0.048633333,0.014],          # CCA-target
                  [0.043066667,0.048266667,0.017]]          # GCCA-target

cue_acc_std = [[0.0373,0.065433333,0.0207],                 # CCA-cue
               [0.0373,0.048833333,0.015066667]]            # GCCA-cue

cog_acc = [[0.501566667,0.7816,0.806],                      # CCA-delay
           [0.501566667,0.761233333,0.7736],                # GCCA-delay
           [0.483166667,0.756166667,0.768233333],           # CCA-target
           [0.483166667,0.766866667,0.780466667],           # GCCA-target
           [0.256, 0.8109, 0.801466667],                    # CCA-cue
           [0.256, 0.816, 0.821966667]]                     # GCCA-cue

cog_acc_std = [[0.045733333,0.046933333,0.025466667],       # CCA-delay
               [0.045733333,0.038733333,0.016466667],       # GCCA-delay
               [0.043066667,0.048633333,0.014],             # CCA-target
               [0.043066667,0.048266667,0.017],             # GCCA-target
               [0.0373, 0.065433333, 0.0207],               # CCA-cue
               [0.0373, 0.048833333, 0.015066667]]          # GCCA-cue


# motor
labels = ['CCA-preparing','GCCA-preparing','CCA-executing','GCCA-executing']
fig,ax = plt.subplots(ncols=1,dpi=300)
for i in range(len(motor_acc)):
    ax.errorbar(np.arange(1,4),motor_acc[i],yerr=motor_acc_std[i],fmt='o-',elinewidth=2,capsize=4,label=labels[i])
x_label = ['unaligned','aligned','within']
ax.set_ylim([0,1])
ax.set_xticks(np.arange(1,4))
ax.set_xticklabels(x_label,fontsize=15)
ax.set_title('Classification accuracy under different conditions', fontdict={'size': 15})
ax.legend(fontsize=12)
ax.set_ylabel('Accuracy', fontdict={'size': 15})
ax.tick_params(axis='y', labelsize=15)
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'motor-MCx-cls-score-all.eps',format='eps', dpi=1000)
plt.clf()
plt.close()


# target
labels = ['CCA-delay','GCCA-delay','CCA-target','GCCA-target']
fig,ax = plt.subplots(ncols=1,dpi=300)
for i in range(len(target_acc)):
    ax.errorbar(np.arange(1,4),target_acc[i],yerr=target_acc_std[i],fmt='o-',elinewidth=2,capsize=4,label=labels[i])
x_label = ['unaligned','aligned','within']
ax.set_ylim([0,1])
ax.set_xticks(np.arange(1,4))
ax.set_xticklabels(x_label,fontsize=15)
ax.set_title('Classification accuracy under different conditions', fontdict={'size': 15})
ax.legend(loc='lower right',fontsize=12)
ax.set_ylabel('Accuracy', fontdict={'size': 15})
ax.tick_params(axis='y', labelsize=15)
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'cog-target-cls-score-all.eps',format='eps', dpi=1000)
plt.clf()
plt.close()


# cue
labels = ['CCA-cue','GCCA-cue']
fig,ax = plt.subplots(ncols=1,dpi=300)
for i in range(len(cue_acc)):
    ax.errorbar(np.arange(1,4),cue_acc[i],yerr=cue_acc_std[i],fmt='o-',elinewidth=2,capsize=4,label=labels[i])
x_label = ['unaligned','aligned','within']
ax.set_ylim([0,1])
ax.set_xticks(np.arange(1,4))
ax.set_xticklabels(x_label,fontsize=15)
ax.set_title('Classification accuracy under different conditions', fontdict={'size': 15})
ax.legend(loc='lower right',fontsize=12)
ax.set_ylabel('Accuracy', fontdict={'size': 15})
ax.tick_params(axis='y', labelsize=15)
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'cog-cue-cls-score-all.eps',format='eps', dpi=1000)
plt.clf()
plt.close()


# cognitive
labels = ['CCA-delay','GCCA-delay','CCA-target','GCCA-target','CCA-cue','GCCA-cue']
fig,ax = plt.subplots(ncols=1,dpi=300)
for i in range(len(cog_acc)):
    ax.errorbar(np.arange(1,4),cog_acc[i],yerr=cog_acc_std[i],fmt='o-',elinewidth=2,capsize=4,label=labels[i])
x_label = ['unaligned','aligned','within']
ax.set_ylim([0.2,0.9])
ax.set_xticks(np.arange(1,4))
ax.set_xticklabels(x_label,fontsize=15)
ax.set_title('Classification accuracy under different conditions', fontdict={'size': 15})
ax.legend(loc='lower right',fontsize=12)
ax.set_ylabel('Accuracy', fontdict={'size': 15})
ax.tick_params(axis='y', labelsize=15)
fig.tight_layout()
plt.savefig(
    'E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/result/figure/'
    'cog-cls-score-all.eps',format='eps', dpi=1000)
plt.clf()
plt.close()