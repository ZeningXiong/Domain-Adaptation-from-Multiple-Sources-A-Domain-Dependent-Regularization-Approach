import os
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from scipy.io import loadmat
from train_fast_dam import train_fast_dam
from return_ind2 import return_ind2

def log_print(file, message, *args):
    """Helper function to log messages to a file."""
    with open(file, 'a') as f:
        f.write(message % args + '\n')

def calc_ap(y_true, y_score):
    """Calculate Average Precision (AP)."""
    return average_precision_score(y_true, y_score)



def main_fast_dam(data, C, N, lambda_L, lambda_D, thr, beta, virtual_label_type, kernel_types, kernel_params):
    """
    Python implementation of the MATLAB `main_fast_dam` function.

    Inputs:
        data : A dictionary containing the dataset and metadata.
        C : SVM regularization parameter.
        N : Number of labeled target samples.
        lambda_L, lambda_D : Regularization parameters.
        thr : Threshold for filtering.
        beta : Parameter for MMD weighting.
        virtual_label_type : Type of virtual label.
        kernel_types : List of kernel types.
        kernel_params : List of kernel parameters for each kernel type.

    Outputs:
        result : A dictionary containing results for each round.
    """
    # Create result directory
    result_dir = f'result_{data["setting"]}'
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f'result_{__name__}.txt')

    log_print(result_file, '<==========  BEGIN @ %s, C = %g, lambda_L = %g, lambda_D = %g, thr = %g, beta = %g, N = %d ============>\n',
              np.datetime64('now'), C, lambda_L, lambda_D, thr, beta, N)

    # Compute kernel matrix
    K = np.dot(data['Xt'], data['Xt'].T)

    

    result = {}
    for r in range(data['nRound']):
        tar_train_index, test_ind = return_ind2(data['perm_tar_index'][r], np.sum(data['yt'] == 1), N)
        tar_test_index = np.setdiff1d(np.arange(len(data['yt'])), tar_train_index)
        all_test_dv = []
        mmd_values = []

        for s in range(len(data['Xs'])):
            if virtual_label_type.endswith('_fr'):
                dv_dir = os.path.join(result_dir, 'decision_values', 'svm_fr', data['domain_names'][s])
                mmd_dir = os.path.join(result_dir, 'mmd_values_fr', data['domain_names'][s])
            elif virtual_label_type.endswith('_s'):
                dv_dir = os.path.join(result_dir, 'decision_values', 'svm_s', data['domain_names'][s])
                mmd_dir = os.path.join(result_dir, 'mmd_values_at', data['domain_names'][s])
            elif virtual_label_type.endswith('_at'):
                dv_dir = os.path.join(result_dir, 'decision_values', 'svm_at', data['domain_names'][s])
                mmd_dir = os.path.join(result_dir, 'mmd_values_at', data['domain_names'][s])

            for kt, kernel_type in enumerate(kernel_types):
                
                for kp, kernel_param in enumerate(kernel_params[kt]):
                    if virtual_label_type.endswith('_s'):
                        dv_file = os.path.join(dv_dir, f'dv_C={C}_{kernel_type}_{kernel_param}.mat')
                    else:
                        dv_file = os.path.join(dv_dir, f'dv_round={r}_N={N}_C={C}_{kernel_type}_{kernel_param}.mat')

                    # Load decision values
                    decision_values = loadmat(dv_file)['decision_values']
                    all_test_dv.append(decision_values)
                    

                    # Load MMD values
                    mmd_file = os.path.join(mmd_dir, f'mmd_{kernel_type}_{kernel_param}.mat')
                    mmd_value = loadmat(mmd_file)['mmd_value']
                    mmd_values.append(mmd_value)
        
        

        mmd_values = np.array(mmd_values)
       

        # Prepare labels and virtual labels
        y = data['yt'].copy()
        y[tar_test_index] = 0
        f_s = all_test_dv

        # Compute gamma_s
        pp = -beta * mmd_values**2
        gamma_s = np.exp(pp)
        gamma_s1 = gamma_s / np.sum(gamma_s)
        g = []
        for i in range(18):
            g.append(gamma_s1[i][0][0])
        gamma_s = g

        # Set parameters for train_fast_dam
        theta1 = lambda_L
        theta2 = lambda_D
        epsilon = 0.1

        # Train the model
        dv, model = train_fast_dam(K, y, f_s, gamma_s, theta1, theta2, thr, f_p=None, lambda_=1, C=C, epsilon=epsilon)

        # Evaluate performance
        ap = calc_ap(data['yt'][tar_test_index], dv[tar_test_index])
        acc = accuracy_score(data['yt'][tar_test_index], np.sign(dv[tar_test_index]))
       

        # Store results
        result[r] = {'ap': ap, 'acc': acc}
        log_print(result_file, '******%g\t%g @ round=%d, C=%g, lambda_L=%g, lambda_D=%g, N = %d, thr=%g\n', 
                  ap, acc, r, C, lambda_L, lambda_D, N, thr)

    log_print(result_file, '<==========  END @ %s, C = %g, lambda_L = %g, lambda_D = %g, thr = %g, beta = %g ============>\n',
              np.datetime64('now'), C, lambda_L, lambda_D, thr, beta)

    return result
