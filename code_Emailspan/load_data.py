import os
import numpy as np
from scipy.io import loadmat

def load_data():
    
    DATASET_DIR = os.path.join('data', 'emailspam')

    
    data = {
        'domain_names': ['U00', 'U01', 'U02', 'U03'],
        'Xs': [],  
        'ys': [], 
        'Xt': None,  
        'yt': None, 
        'tar_train_index': [],  
        'tar_test_index': [],  
        'tar_background_index': [],  
        'nRound': 10  
    }

    
    for i in range(len(data['domain_names']) - 1):
        file_path = os.path.join(DATASET_DIR, f'data_{i+1}.mat')
        t = loadmat(file_path)
        data['Xs'].append(t['features']) 
        data['ys'].append(t['labels'].ravel())  

    
    file_path = os.path.join(DATASET_DIR, 'data_4.mat')
    t = loadmat(file_path)
    data['Xt'] = t['features'] 
    data['yt'] = t['labels'].ravel()  

    
    
    pos_index = np.where(data['yt'] == 1)[0]
    neg_index = np.where(data['yt'] == -1)[0]
    pos_index = np.random.permutation(pos_index)
    neg_index = np.random.permutation(neg_index)

    
    tar_ind_dir = os.path.join(DATASET_DIR, 'tar_ind2')
    for i in range(1, 11):  # 1-10
        file_path = os.path.join(tar_ind_dir, f'{i}.mat')
        t = loadmat(file_path)
        data['tar_train_index'].append(t['tar_training_ind'].ravel() - 1)  
        data['tar_test_index'].append(t['test_ind'].ravel() - 1)
        data['tar_background_index'].append(t['tar_background_ind'].ravel() - 1)

    return data

