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

# load aligned latent dynamics
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/motor/'
          'motor_latent_dynamics_aligned_pca10_full.pickle', 'rb') as f:
    latent_dynamics_aligned = pickle.load(f)
with open('E:\MSc Project\monkey dataset\motor/motor_CEBRA_data_aligned_pca10.pickle', 'rb') as f:
    data = pickle.load(f)

init_comp = latent_dynamics_aligned[0][3][0].shape[-1]
cls_scores = []
for i,dyna in enumerate(latent_dynamics_aligned):
    id1,id2,id3,AllDyna = dyna
    id_all = [id1,id2,id3]
    n_targets, n_trial1, n_time1, _ = data[i][3][0].shape
    AllDyna = np.array([AllDyna[ii].reshape((n_targets,n_trial1,n_time1,10)) for ii in range(len(AllDyna))])
    # AllDyna = dt.add_history_to_data_array(AllDyna, MAX_HISTORY)

    n_comp = AllDyna.shape[4]
    scores = [[],[],[]]
    for subj in range(AllDyna.shape[0]):
        X_train_tmp = AllDyna[subj].reshape((-1,n_time1 * n_comp))

        AllTar_train = np.repeat(np.arange(8), n_trial1) # labels
        trial_index1 = np.arange(len(AllTar_train))
        while ((all_id_sh := rng.permutation(trial_index1)) == trial_index1).all():
            continue
        trial_index1 = all_id_sh
        X_train, Y_train = X_train_tmp[trial_index1, :], AllTar_train[trial_index1]

        classifier = classifier_model(**classifier_params)
        classifier.fit(X_train, Y_train) # train classifier

        index_tmp = np.delete([0,1,2],subj) # exclude subject for training
        X_test1 = AllDyna[index_tmp[0]]
        X_test2 = AllDyna[index_tmp[1]]

        X_test1 = X_test1.reshape((-1, n_time1 * n_comp))
        X_test2 = X_test2.reshape((-1, n_time1 * n_comp))
        AllTar1 = np.repeat(np.arange(8), n_trial1)

        rng.shuffle(trial_index1)
        X1_test, Y1_test = X_test1[trial_index1, :], AllTar1[trial_index1]
        X2_test, Y2_test = X_test2[trial_index1, :], AllTar1[trial_index1]

        # test the decoder
        scores[subj].append(classifier.score(X1_test, Y1_test)) # test classifier
        scores[subj].append(classifier.score(X2_test, Y2_test))
    cls_scores.append((id1, id2, id3, scores))
    print('Pair '+str(i+1)+' done')
warnings.filterwarnings("default")
# change with your own saving path
with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result/'
          'motor_MCx_CEBRA_cls_scores_pca10.pickle', 'wb') as f:
    pickle.dump(cls_scores, f)

file = open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu/CEBRA/result/'
            'motor_MCx_CEBRA_cls_scores_pca10.txt','w')
for fp in cls_scores:
    file.write(str(fp))
    file.write('\n')
file.close()
