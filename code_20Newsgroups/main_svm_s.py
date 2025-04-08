import os
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, average_precision_score
import datetime
from return_ind2 import return_ind2
from scipy.sparse import csr_matrix
from calc_kernel_S import calc_kernel_S

def main_svm_s(data, C, N, kernel_types, kernel_params):
    
   
    result_dir = f'result_{data['setting']}'
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f'result_{os.path.basename(__file__)}.txt')
    
    
    def log_print(message, *args):
        with open(result_file, 'a') as f:
            f.write(message % args + '\n')
    
    log_print('<==========  BEGIN @ %s, C = %g ============>\n', 
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), C)

    
    for s in range(len(data['Xs'])):
        domain_name = data['domain_names'][s]
        dv_dir = os.path.join(result_dir, 'decision_values', 'svm_s', domain_name)
        os.makedirs(dv_dir, exist_ok=True)
        
        Xs = data['Xs'][s]
        ys = data['ys'][s].ravel()  
        
        
        S = np.dot(Xs, Xs.T).toarray()
        T = np.dot(data['Xt'], Xs.T).toarray()
        
        for kt, kernel_type in enumerate(kernel_types):
            for kp, kernel_param in enumerate(kernel_params[kt]):
               
                K_train = calc_kernel_S(kernel_type, kernel_param, S)
                K_test = calc_kernel_S(kernel_type, kernel_param, T)

                
                
                
                for r in range(data['nRound']):
                    
                    tar_train_idx = return_ind2(data['perm_tar_index'][r], 
                                              np.sum(data['yt'] == 1), N)
                    
                    
                    dv_file = os.path.join(dv_dir, 
                                         f'dv_C={C}_{kernel_type}_{ kernel_param}.mat')
                    
                    if not os.path.exists(dv_file):
                        
                        model = SVC(C=C, kernel='precomputed', probability=False)
                        model.fit(K_train, ys)
                        
                        
                        decision_values = model.decision_function(K_test)
                        
                        
                        savemat(dv_file, {'decision_values': decision_values})

    
    result = {}
    for r in range(data['nRound']):
        result[r] = {'ap_sigmoid': None, 'acc_sigmoid': None,
                    'ap_no_sigmoid': None, 'acc_no_sigmoid': None}
        
        
        tar_train_idx, test_ind = return_ind2(data['perm_tar_index'][r], 
                                  np.sum(data['yt'] == 1), N)
        
        tar_test_idx = np.setdiff1d(np.arange(len(data['yt'])), tar_train_idx)
        
        all_test_dv = []
        for s in range(len(data['Xs'])):
            domain_name = data['domain_names'][s]
            dv_dir = os.path.join(result_dir, 'decision_values', 'svm_s', domain_name)
            
            for kt, kernel_type in enumerate(kernel_types):
                for kp, kernel_param in enumerate(kernel_params[kt]):
                    
                    dv_file = os.path.join(dv_dir, 
                                         f'dv_C={C}_{kernel_type}_{kernel_param}.mat')
                    
                    
                    data_dv = loadmat(dv_file)
                    decision_values = data_dv['decision_values'].ravel()
                    
                    
                    y_true = data['yt'][tar_test_idx].ravel()
                    y_score = decision_values[tar_test_idx]
                   
                    all_test_dv.append(y_score)
                    ap = average_precision_score(y_true, y_score)
                    acc = accuracy_score(y_true, np.sign(y_score))
                    
                    log_print('%g\t%g @ round=%d, C=%g, kernel=%s, kernel_param=%g, N=%d, %s\n', 
                             ap, acc, r, C, kernel_type, kernel_param,N, domain_name)
                    
        
        
        if len(all_test_dv) == 0:
            raise ValueError("all_test_dv is empty")
        else:
            all_test_dv = np.column_stack(all_test_dv)
        
        # Sigmoid 
        
        dv_sigmoid = np.mean(1 / (1 + np.exp(-all_test_dv)), axis=1)
        
        ap = average_precision_score(data['yt'][tar_test_idx], dv_sigmoid[tar_test_idx-2*N])
        acc = accuracy_score(data['yt'][tar_test_idx], np.sign(dv_sigmoid[tar_test_idx-2*N]))
        result[r]['ap_sigmoid'] = ap
        result[r]['acc_sigmoid'] = acc
        log_print('SIGMIOD %g\t%g @ round=%d, C=%g\n', ap, acc, r, C)
        
        # average
        dv_mean = np.mean(all_test_dv, axis=1)
        ap = average_precision_score(data['yt'][tar_test_idx], dv_mean[tar_test_idx-2*N])
        acc = accuracy_score(data['yt'][tar_test_idx], np.sign(dv_mean[tar_test_idx-2*N]))
        result[r]['ap_no_sigmoid'] = ap
        result[r]['acc_no_sigmoid'] = acc
        log_print('NO_SIGMIOD %g\t%g @ round=%d, C=%g\n', ap, acc, r, C)
    
    
    
    return result
