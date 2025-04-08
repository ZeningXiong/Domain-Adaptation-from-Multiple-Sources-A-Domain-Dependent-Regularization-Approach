import os
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, average_precision_score
import datetime
from calc_kernel_S import calc_kernel_S
from scipy.sparse import vstack

def main_svm_fr(data, C, kernel_types, kernel_params):
   
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'svm_fr', f'result_{os.path.basename(__file__)}.txt')
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    
    def log_print(message, *args):
        with open(result_file, 'a') as f:
            f.write(message % args + '\n')

    log_print('<==========  BEGIN @ %s, C = %g ============>\n', 
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), C)

    
    for s in range(len(data['Xs'])):
        domain_name = data['domain_names'][s]
        dv_dir = os.path.join(result_dir, 'svm_fr', 'decision_values', domain_name)
        os.makedirs(dv_dir, exist_ok=True)

        Xs = data['Xs'][s]
        ys = data['ys'][s]

        
        X_combined = vstack((data['Xt'], Xs))
        y_combined = np.concatenate((data['yt'], ys))
        X_combined = X_combined.toarray()

        
        size_Xt = data['Xt'].shape[0]
        size_Xs = Xs.shape[0]
        src_index = np.arange(size_Xs) + size_Xt 
        tar_index = np.arange(size_Xt)            

       
        S = np.dot(X_combined, X_combined.T)

        for kt, kernel_type in enumerate(kernel_types):
            for kp, kernel_param in enumerate(kernel_params[kt]):
                
                K = calc_kernel_S(kernel_type, kernel_param, S)

                
                K[src_index[:, None], src_index] *= 2  
                K[tar_index[:, None], tar_index] *= 2  

                for r in range(data['nRound']):
                    
                    tar_train_index = data['tar_train_index'][r]
                    tar_test_index = data['tar_test_index'][r]
                    train_index = np.concatenate([src_index, tar_train_index])
                    
                    

                    
                    dv_file = os.path.join(dv_dir, f'dv_round={r}_C={C}_{kernel_type}_{kernel_param}.mat')

                    if os.path.exists(dv_file):
                        
                        data_dv = loadmat(dv_file)
                        decision_values = data_dv['decision_values'].ravel()
                    else:
                        
                        model = SVC(C=C, kernel='precomputed', probability=False)
                        model.fit(K[train_index[:, None], train_index], y_combined[train_index])

                        
                        decision_values = model.decision_function(K[tar_index[:, None], train_index])

                        
                        savemat(dv_file, {'decision_values': decision_values})

    
    result = {}
    for r in range(data['nRound']):
        result[r] = {'ap_sigmoid': None, 'acc_sigmoid': None,
                     'ap_no_sigmoid': None, 'acc_no_sigmoid': None}

        
        tar_test_index = data['tar_test_index'][r]

        all_test_dv = []
        for s in range(len(data['Xs'])):
            domain_name = data['domain_names'][s]
            dv_dir = os.path.join(result_dir, 'svm_fr', 'decision_values', domain_name)

            for kt, kernel_type in enumerate(kernel_types):
                for kp, kernel_param in enumerate(kernel_params[kt]):
                    
                    dv_file = os.path.join(dv_dir, f'dv_round={r}_C={C}_{kernel_type}_{kernel_param}.mat')
                    data_dv = loadmat(dv_file)
                    decision_values = data_dv['decision_values'].ravel()

                    
                    y_true_test = data['yt'][tar_test_index]
                    y_score_test = decision_values[tar_test_index]

                    ap = average_precision_score(y_true_test, y_score_test)
                    acc = accuracy_score(y_true_test, np.sign(y_score_test))

                    log_print('%g\t%g @ round=%d, C=%g, kernel=%s, kernel_param=%g, %s\n', 
                              ap, acc, r, C, kernel_type, kernel_param, domain_name)
                    all_test_dv.append(y_score_test)

        #  (n_samples, n_models)
        all_test_dv = np.column_stack(all_test_dv)

        # Sigmoid 
        dv_sigmoid = np.mean(1 / (1 + np.exp(-all_test_dv)), axis=1)
        
        ap = average_precision_score(data['yt'][tar_test_index], dv_sigmoid[list(range(len(tar_test_index)))])
        acc = accuracy_score(data['yt'][tar_test_index], np.sign(dv_sigmoid[list(range(len(tar_test_index)))]))
        result[r]['ap_sigmoid'] = ap
        result[r]['acc_sigmoid'] = acc
        log_print('SIGMIOD %g\t%g @ round=%d, C=%g\n', ap, acc, r, C)

       
        dv_mean = np.mean(all_test_dv, axis=1)
        ap = average_precision_score(data['yt'][tar_test_index], dv_mean[list(range(len(tar_test_index)))])
        acc = accuracy_score(data['yt'][tar_test_index], np.sign(dv_mean[list(range(len(tar_test_index)))]))
        result[r]['ap_no_sigmoid'] = ap
        result[r]['acc_no_sigmoid'] = acc
        log_print('NO_SIGMIOD %g\t%g @ round=%d, C=%g\n', ap, acc, r, C)

    

    return result



