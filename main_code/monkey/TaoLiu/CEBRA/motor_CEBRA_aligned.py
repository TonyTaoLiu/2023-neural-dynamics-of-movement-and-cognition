import sys, os, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cebra import CEBRA
import pickle

latent_dynamics_full = []
latent_dynamics_avg = []
max_iterations = 5000 # number of iterations
output_dim = 10 # dimension of latent embedding

with open('E:\MSc Project\monkey dataset\motor/motor_CEBRA_data_aligned_pca10.pickle', 'rb') as f:
      data = pickle.load(f)
with open('E:\MSc Project\monkey dataset\motor/motor_CEBRA_label_aligned_pca10.pickle', 'rb') as f:
      label = pickle.load(f)

for group_num in range(len(data)):
  if group_num > 64: # training takes long time, allows to divide training into several parts
    id1,id2,id3,AllData = data[group_num]
    id1,id2,id3,AllVel = label[group_num]
    data1 = AllData[0].reshape([-1,AllData[0].shape[-1]])
    data2 = AllData[1].reshape([-1,AllData[1].shape[-1]])
    data3 = AllData[2].reshape([-1,AllData[2].shape[-1]])

    label1 = AllVel[0].reshape([-1,AllVel[0].shape[-1]])
    label2 = AllVel[1].reshape([-1,AllVel[1].shape[-1]])
    label3 = AllVel[2].reshape([-1,AllVel[2].shape[-1]])

    data_input = [data1,data2,data3]
    label_input = [label1,label2,label3]

    multi_embeddings = []

    # Multisession training
    multi_cebra_model = CEBRA(model_architecture='offset10-model',
                              batch_size=512,
                              learning_rate=1e-4,
                              temperature=1,
                              output_dimension=output_dim,
                              max_iterations=max_iterations,
                              distance='cosine',
                              conditional='time_delta',
                              device='cuda',
                              verbose=True,
                              time_offsets=1)

    multi_cebra_model.fit(data_input,label_input) # supervised learning with velocity as label

    for ii, X in enumerate(data_input):
      multi_embeddings.append(multi_cebra_model.transform(X, session_id=ii)) # transformation

    latent_dynamics_full.append((id1,id2,id3,multi_embeddings))
    # change with your own saving path
    # save separately, allows dividing training into several parts
    with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor/motor_latent_dynamics_aligned_pca10_full'+str(group_num)+'.pickle','wb') as f:
      pickle.dump(latent_dynamics_full[-1], f)

    for i in range(8):
        target_embeddings = []
        for j in range(len(multi_embeddings)):
            trial_avg = multi_embeddings[j].reshape(8,AllData[0].shape[1],AllData[0].shape[2],output_dim)[i,...].mean(axis=0)
            # trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
            target_embeddings.append(trial_avg)
        latent_dynamics_avg.append((id1,id2,id3,i,target_embeddings))

    # change with your own saving path
    # save separately, allows dividing training into several parts
    with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor/motor_latent_dynamics_aligned_pca10_avg_'+str(group_num)+'.pickle','wb') as f:
      pickle.dump(latent_dynamics_avg[-8:], f)

# with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor/motor_latent_dynamics_aligned_full.pickle','wb') as f:
#     pickle.dump(latent_dynamics_full, f)
# with open('E:\MSc Project\code/2022-preserved-dynamics-main\monkey\TaoLiu\CEBRA/result\motor/motor_latent_dynamics_aligned_avg.pickle','wb') as f:
#     pickle.dump(latent_dynamics_avg, f)