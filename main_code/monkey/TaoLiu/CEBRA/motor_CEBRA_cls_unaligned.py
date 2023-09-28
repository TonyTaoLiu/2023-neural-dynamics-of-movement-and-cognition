import os, sys, pathlib
import pickle
import logging, warnings
logging.basicConfig(level=logging.ERROR)
import numpy as np
from monkey.defs import *
from sklearn.naive_bayes import GaussianNB
from tools.lstm import *
from tools import dataTools as dt


classifier_model = GaussianNB
classifier_params = {}

# load unaligned latent dynamics
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_latent_dynamics_unaligned_pca10_full.pickle', 'rb') as f:
    latent_dynamics_unaligned = pickle.load(f)
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_input_labels_unaligned_pca10.pickle', 'rb') as f:
    labels = pickle.load(f)

_, n_targets, n_trial1, n_time1, _ = labels[0].shape
init_comp = latent_dynamics_unaligned[0][1][0].shape[-1]
cls_scores = []
num = 1
for id1,ld1 in enumerate(latent_dynamics_unaligned[0][1]):
    for id2,ld2 in enumerate(latent_dynamics_unaligned[1][1]):
        for id3,ld3 in enumerate(latent_dynamics_unaligned[2][1]):
            AllDyna = [ld1,ld2,ld3]
            AllDyna = np.array([AllDyna[ii].reshape((n_targets,n_trial1,n_time1,init_comp)) for ii in range(len(AllDyna))])
    # AllDyna = dt.add_history_to_data_array(AllDyna, MAX_HISTORY)

            n_comp = AllDyna.shape[4]
            scores = [[] for num in range(AllDyna.shape[0])]
            for subj in range(AllDyna.shape[0]):
                X_train_tmp = AllDyna[subj].reshape((-1,n_time1 * n_comp))

    # X1 = AllDyna[0].reshape((-1, n_time1 * n_comp))
                AllTar_train = np.repeat(np.arange(8), n_trial1) # labels
                trial_index1 = np.arange(len(AllTar_train))
                while ((all_id_sh := rng.permutation(trial_index1)) == trial_index1).all():
                    continue
                trial_index1 = all_id_sh
                X_train, Y_train = X_train_tmp[trial_index1, :], AllTar_train[trial_index1]

                classifier = classifier_model(**classifier_params)
                classifier.fit(X_train, Y_train) # train classifier

                index_all = np.arange(AllDyna.shape[0])
                index_tmp = np.delete(index_all,subj) # exclude subject for training
                for iii in range(index_tmp.shape[0]): # test on rest subjects
                    X_test0 = AllDyna[index_tmp[iii]]
                    X_test0 = X_test0.reshape((-1, n_time1 * n_comp))
                    AllTar1 = np.repeat(np.arange(8), n_trial1)

                    rng.shuffle(trial_index1)
                    X_test, Y_test = X_test0[trial_index1, :], AllTar1[trial_index1]

                    # test the decoder
                    scores[subj].append(classifier.score(X_test, Y_test)) # test classifier
            cls_scores.append((id1,id2,id3, scores))
            print('Pair '+str(num)+' done')
            num = num + 1
warnings.filterwarnings("default")
# change with your own saving path
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result/'
          'motor_MCx_CEBRA_cls_scores_pca10_unaligned.pickle', 'wb') as f:
    pickle.dump(cls_scores, f)

file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
            'motor_MCx_CEBRA_cls_scores_pca10_unaligned.txt','w')
for fp in cls_scores:
    file.write(str(fp))
    file.write('\n')
file.close()