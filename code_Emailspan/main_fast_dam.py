import os
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, average_precision_score
import datetime
from train_fast_dam import train_fast_dam

def main_fast_dam(data, C, lambda_L, lambda_D, thr, beta, virtual_label_type, kernel_types, kernel_params):
    
    
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'fast_dam', f'result_{os.path.basename(__file__)}.txt')
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

   
    def log_print(message, *args):
        with open(result_file, 'a') as f:
            f.write(message % args + '\n')

    log_print('<==========  BEGIN @ %s, C = %g, lambda_L = %g, lambda_D = %g, thr = %g, beta = %g ============>\n',
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), C, lambda_L, lambda_D, thr, beta)

    
    K = np.dot(data['Xt'], data['Xt'].T)

    
    result = {}

  
    for r in range(data['nRound']):
        tar_train_index = data['tar_train_index'][r]
        tar_test_index = data['tar_test_index'][r]
        all_test_dv = []
        mmd_values = []

        
        for s in range(len(data['Xs'])):
            domain_name = data['domain_names'][s]

            
            if virtual_label_type.endswith('_fr'):
                dv_dir = os.path.join(result_dir, 'svm_fr', 'decision_values', domain_name)
                mmd_dir = os.path.join(result_dir, 'mmd_values_fr', domain_name)
            elif virtual_label_type.endswith('_s'):
                dv_dir = os.path.join(result_dir, 'svm_s', 'decision_values', domain_name)
                mmd_dir = os.path.join(result_dir, 'mmd_values_at', domain_name)
            elif virtual_label_type.endswith('_st'):
                dv_dir = os.path.join(result_dir, 'svm_at', 'decision_values', domain_name)
                mmd_dir = os.path.join(result_dir, 'mmd_values_at', domain_name)
            else:
                raise ValueError("Not support！")

            
            for kt, kernel_type in enumerate(kernel_types):
                for kp in kernel_params[kt]:
                    
                    dv_file = os.path.join(dv_dir, f'dv_round={r}_C={C}_{kernel_type}_{kp}.mat')
                    
                    data_dv = loadmat(dv_file)
                    decision_values = data_dv['decision_values'].ravel()
                    
                    
                    mmd_file = os.path.join(mmd_dir, f'mmd_{kernel_type}_{kp}.mat')
                    
                    data_mmd = loadmat(mmd_file)
                    
                    mmd_value = data_mmd['mmd_value'][0, 0]
                    

                    
                    mmd_values.append(mmd_value)
                    all_test_dv.append(decision_values)

       
        y = data['yt'].copy()
        y[tar_test_index] = 0  
        
        f_s = np.column_stack(all_test_dv)

        
        mmd_values = np.array(mmd_values)
        gamma_s = np.exp(-beta * mmd_values**2)
        gamma_s = gamma_s / np.sum(gamma_s)
        

        
        theta1 = lambda_L
        theta2 = lambda_D
        epsilon = 0.1
        dv, model = train_fast_dam(K, y, f_s, gamma_s, theta1, theta2, thr, f_p=None, lambda_=1.0, C=C, epsilon=epsilon)

        
        ap = average_precision_score(data['yt'][tar_test_index], dv[tar_test_index])
        acc = accuracy_score(data['yt'][tar_test_index], np.sign(dv[tar_test_index]))

        
        result[r] = {'ap': ap, 'acc': acc}
        log_print('******%g\t%g @ round=%d, C=%g, lambda_L=%g, lambda_D=%g, thr=%g\n',
                  ap, acc, r, C, lambda_L, lambda_D, thr)
    

    

    return result

