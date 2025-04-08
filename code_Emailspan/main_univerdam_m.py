import os
import numpy as np
from scipy.io import loadmat, savemat
import datetime
import subprocess
from scipy.sparse import vstack as sparse_vstack, issparse
from sklearn.metrics import accuracy_score, average_precision_score

def main_univerdam(data, C, lambda_L, lambda_D1, lambda_D2, thr, beta, virtual_label_type, kernel_types, kernel_params):

    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'univerdam', f'result_{os.path.basename(__file__)}.txt')
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    

    def log_print(message, *args):
        with open(result_file, 'a') as f:
            f.write(message % args + '\n')
            
    def debug_print(message, *args):
        print(message % args)

    lambda_D = lambda_D1

    log_print('<==========  BEGIN @ %s, C = %g, lambda_L = %g, lambda_D = %g, thr = %g, beta = %g ============>',
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), C, lambda_L, lambda_D, thr, beta)

    X_list = []
    y_list = []
    
    if issparse(data['Xt']):
        X_list.append(data['Xt'].tocsr())
    else:
        X_list.append(data['Xt'])
    y_list.append(data['yt'])

    debug_print("Xt shape: %s", str(data['Xt'].shape))
    

    domain_index = [np.arange(len(data['yt']))]
    offset = len(data['yt'])
    

    for s in range(len(data['Xs'])):
        if issparse(data['Xs'][s]):
            X_list.append(data['Xs'][s].tocsr())
        else:
            X_list.append(data['Xs'][s])
        y_list.append(data['ys'][s])
        

        debug_print("Xs[%d] shape: %s", s, str(data['Xs'][s].shape))
        
        domain_index.append(np.arange(len(data['ys'][s])) + offset)
        offset += len(data['ys'][s])
    

    if issparse(X_list[0]):
        X = sparse_vstack(X_list)

        debug_print("Calculating kernel matrix...")
        K = X.dot(X.T).toarray()
        debug_print("Kernel matrix shape: %s", str(K.shape))
    else:
        X = np.vstack(X_list)
        K = np.dot(X, X.T)
    
    y = np.hstack(y_list)
    tar_index = domain_index[0]
    src_index = np.hstack(domain_index[1:]) if len(domain_index) > 1 else np.array([], dtype=int)
    

    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    

    bat_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_matlab.bat")
    if not os.path.exists(bat_file):
        with open(bat_file, 'w') as f:
            f.write('@echo off\n')
            f.write('set MATLAB_PATH=F:\\MATLAB\\bin\\win64\\MATLAB.exe\n')
            f.write('set LIBSVM_PATH=F:/MATLAB/libsvm-3.35/libsvm-3.35/matlab\n')
            f.write('set DATA_FILE=%1\n')
            f.write('REM Remove quotes from the path\n')
            f.write('set DATA_FILE=%DATA_FILE:"=%\n\n')
            f.write('REM Create a temporary MATLAB script\n')
            f.write('echo addpath(\'%LIBSVM_PATH%\'); > temp_script.m\n')
            f.write('echo disp(\'LIBSVM path added\'); >> temp_script.m\n')
            f.write('echo train_svm_model(\'%DATA_FILE%\'); >> temp_script.m\n\n')
            f.write('REM Execute MATLAB\n')
            f.write('"%MATLAB_PATH%" -batch "run(\'temp_script.m\')"')
        debug_print("MATLAB batch file created: %s", bat_file)

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
                raise ValueError("Unsupported virtual label type!")

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

        f_s = np.column_stack(all_test_dv)
        mmd_values = np.array(mmd_values)
        gamma_s = np.exp(-beta * mmd_values**2)
        gamma_s = gamma_s / np.sum(gamma_s)

        virtual_label = np.dot(f_s, gamma_s)

        tilde_y = y.copy()
        tilde_y[src_index] = 0
        tilde_y[tar_test_index] = virtual_label[tar_test_index]

        add_kernel = np.ones(len(tilde_y))
        add_kernel[src_index] = 1 / lambda_D2
        add_kernel[tar_train_index] = 1 / lambda_L
        add_kernel[tar_test_index] = 1 / lambda_D1 / np.sum(gamma_s)

        ind = np.abs(virtual_label[tar_test_index]) < thr
        v_ind = np.hstack([src_index, tar_train_index, np.setdiff1d(tar_test_index, tar_test_index[ind])])

        epsilon = 0.1
        
        temp_file = os.path.join(temp_dir, f"svm_data_round{r}.mat")
        debug_print("Saving data to %s", temp_file)

        debug_print("tilde_y length: %d", len(tilde_y))
        debug_print("v_ind length: %d, min: %d, max: %d", 
                   len(v_ind), v_ind.min() if len(v_ind) > 0 else 0, v_ind.max() if len(v_ind) > 0 else 0)

        savemat(temp_file, {
            'tilde_y': tilde_y,
            'v_ind': v_ind + 1,  
            'K': K,
            'add_kernel': add_kernel,
            'tar_index': tar_index + 1,  
            'C': C,
            'epsilon': epsilon
        })
        
        debug_print("Round %d/%d: Calling MATLAB for SVM training...", r+1, data['nRound'])
        temp_file_unix = temp_file.replace("\\", "/")
        

        try:
            process = subprocess.run([bat_file, temp_file_unix], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            debug_print("MATLAB execution completed")
        except subprocess.CalledProcessError as e:
            debug_print("MATLAB execution failed, return code: %d", e.returncode)
            debug_print("MATLAB standard output: %s", e.stdout.decode('utf-8', errors='ignore'))
            debug_print("MATLAB error output: %s", e.stderr.decode('utf-8', errors='ignore'))
            raise RuntimeError("MATLAB execution failed")
                
        result_data = loadmat(temp_file)
        
        if 'dv' not in result_data:
            raise RuntimeError("MATLAB training failed: Decision values not found")
            
        dv = result_data['dv'].flatten()
        debug_print("Decision values shape: %s", str(dv.shape))
        
        ap = average_precision_score(y[tar_test_index], dv[tar_test_index])
        acc = np.mean(y[tar_test_index] == np.sign(dv[tar_test_index]))

        result[r] = {'ap': ap, 'acc': acc}

        log_print('******%g\t%g @ round=%d, C=%g, lambda_L=%g, lambda_D=%g, thr=%g',
                  ap, acc, r, C, lambda_L, lambda_D, thr)
    
    try:
        for r in range(data['nRound']):
            temp_file = os.path.join(temp_dir, f"svm_data_round{r}.mat")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_script.m")):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_script.m"))
    except:
        pass

    log_print('<==========  END @ %s, C = %g, lambda_L = %g, lambda_D = %g, thr = %g, beta = %g ============>',
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), C, lambda_L, lambda_D, thr, beta)

    return result
