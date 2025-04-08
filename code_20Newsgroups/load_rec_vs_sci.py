import os
import scipy.io

def load_rec_vs_sci():
    
    DATASET_DIR = os.path.join('data', '20Newsgroups', 'rec_vs_sci')
    
    
    data = {
        'Xs': [],  
        'ys': [], 
        'domain_names': [],  
        'Xt': None, 
        'yt': None,  
        'nRound': 10,  
        'perm_tar_index': []  
    }

    
    print('rec vs sci or 2 3 4 5 vs 8 9 10 11')

    
    src_data_path = os.path.join(DATASET_DIR, 'src_data.mat')
    src_data = scipy.io.loadmat(src_data_path)

    
    data['Xs'].append(src_data['src1_features'])
    data['ys'].append(src_data['src1_labels'].flatten())
    data['domain_names'].append('2vs8')

   
    data['Xs'].append(src_data['src2_features'])
    data['ys'].append(src_data['src2_labels'].flatten())
    data['domain_names'].append('3vs9')

    
    data['Xs'].append(src_data['src3_features'])
    data['ys'].append(src_data['src3_labels'].flatten())
    data['domain_names'].append('4vs10')

    
    tar_data_path = os.path.join(DATASET_DIR, 'tar_data.mat')
    tar_data = scipy.io.loadmat(tar_data_path)

    data['Xt'] = tar_data['tar_features']
    data['yt'] = tar_data['tar_labels'].flatten()
    data['domain_names'].append('5vs11')

    
    tar_ind2_dir = os.path.join(DATASET_DIR, 'tar_ind2')
    for r in range(1, data['nRound'] + 1):
        tar_index_path = os.path.join(tar_ind2_dir, f'{r}_all.mat')
        tar_index_data = scipy.io.loadmat(tar_index_path)
        data['perm_tar_index'].append(tar_index_data['tar_training_ind'].flatten())

    return data
