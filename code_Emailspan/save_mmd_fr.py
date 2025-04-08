import os
import numpy as np
from scipy.io import savemat, loadmat
from load_data import load_data
from calc_kernel_S import calc_kernel_S
from scipy.sparse import vstack 
def save_mmd_fr():
    
    
    data = load_data()
    
    
    kernel_types = ['linear', 'poly']
    kernel_params = [
        [0],                
        [1.1, 1.2, 1.3, 1.4, 1.5]  
    ]
    
    
    
    result_dir = f'results'
    os.makedirs(result_dir, exist_ok=True)
    
   
    for s in range(len(data['Xs'])):
        Xs = data['Xs'][s]
        domain_name = data['domain_names'][s]
        
        
        mmd_dir = os.path.join(result_dir, 'mmd_values_fr', domain_name)
        os.makedirs(mmd_dir, exist_ok=True)
        
        
        X_combined = vstack((data['Xt'], Xs))
        
        
        X_combined = X_combined.toarray()
    
        
        
        size_Xt = data['Xt'].shape[0]
        size_Xs = Xs.shape[0]
        src_index = np.arange(size_Xs) + size_Xt  
        tar_index = np.arange(size_Xt)            
        
        
        s = np.zeros(size_Xt + size_Xs)
        s[src_index] = 1.0 / size_Xs
        s[tar_index] = -1.0 / size_Xt
        
        
        
        S = np.dot(X_combined, X_combined.T)
        
        
        for kt, kernel_type in enumerate(kernel_types):
            for kp in kernel_params[kt]:
               
                mmd_file = os.path.join(mmd_dir, f'mmd_{kernel_type}_{kp}.mat')
                
                
                if os.path.exists(mmd_file):
                    data_mmd = loadmat(mmd_file)
                    mmd_value = data_mmd['mmd_value'][0,0]
                else:
                    
                    K = calc_kernel_S(kernel_type, kp, S)
                    
                    
                    mmd_value = np.dot(s.T, np.dot(K, s))
                    
                    
                    savemat(mmd_file, {'mmd_value': mmd_value})
                
                print(f'Domain: {domain_name}, Kernel: {kernel_type}({kp}), MMD = {mmd_value:.4f}')



save_mmd_fr()

