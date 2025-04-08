import os
import numpy as np
import subprocess
from sklearn.metrics import average_precision_score
from scipy.io import loadmat, savemat
from scipy.sparse import vstack as sparse_vstack, csr_matrix
from return_ind2 import return_ind2

def log_print(file, message, *args):
    
    with open(file, 'a') as f:
        f.write(message % args + '\n')

def calc_ap(y_true, y_score):
    
    return average_precision_score(y_true, y_score)

def main_univerdam_m(data, C, N, lambda_L, lambda_D1, lambda_D2, thr, beta, virtual_label_type, kernel_types, kernel_params):

    result_dir = f"result_{data['setting']}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"result_{__name__}.txt")
    
    log_print(result_file,
              "<==========  BEGIN @ %s, C = %g, lambda_L = %g, lambda_D1 = %g, lambda_D2 = %g, thr = %g, beta = %g ===========>\n",
              np.datetime64('now'), C, lambda_L, lambda_D1, lambda_D2, thr, beta)
    
    X_list = [data['Xt'].tocsr()]
    y_list = [data['yt']]

    print(f"Xt shape: {data['Xt'].shape}")
    for s in range(len(data['Xs'])):
        print(f"Xs[{s}] shape: {data['Xs'][s].shape}")

    domain_index = [np.arange(len(data['yt']))]
    offset = len(data['yt'])
    for s in range(len(data['Xs'])):
        X_list.append(data['Xs'][s].tocsr())
        y_list.append(data['ys'][s])
        current_indices = np.arange(len(data['ys'][s])) + offset
        domain_index.append(current_indices)
        offset += len(data['ys'][s])
        
    X_all = sparse_vstack(X_list)
    print(f"X_all shape: {X_all.shape}")
    y_all = np.hstack(y_list)
    tar_index = domain_index[0]
    src_index = np.hstack(domain_index[1:]) if len(domain_index) > 1 else np.array([], dtype=int)

    K = X_all.dot(X_all.T).toarray()
    
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    matlab_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_svm_model.m")
    if not os.path.exists(matlab_script):

        with open(matlab_script, 'w') as f:
            f.write("""function train_svm_model(data_file)
    % Load data
    load(data_file);
    
    try
        % Train the SVM model
        fprintf('Executing svmtrain...\\n');
        model = svmtrain(tilde_y, [(1:length(v_ind))', K_v], sprintf('-s 3 -c %g -t 4 -q -p %g', C, epsilon));
        
        % Map the support vectors to the original indices
        model.SVs = v_ind(model.SVs);
        
        % Compute decision values
        fprintf('Calculating decision values...\\n');
        dv = K(tar_index, model.SVs) * model.sv_coef - model.rho;
        
        % Save results
        save(data_file, 'model', 'dv', '-append');
        fprintf('SVM training completed\\n');
    catch ME
        fprintf('Error: %s\\n', ME.message);
        exit(1);
    end
    exit(0);
end""")
            print(f"MATLAB script created: {matlab_script}")
    
    result = {}
    for r in range(data['nRound']):
        tar_train_index, _ = return_ind2(data['perm_tar_index'][r], np.sum(data['yt'] == 1), N)
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
            elif virtual_label_type.endswith('_st'):
                dv_dir = os.path.join(result_dir, 'decision_values', 'svm_at', data['domain_names'][s])
                mmd_dir = os.path.join(result_dir, 'mmd_values_at', data['domain_names'][s])
            

            for kt, kernel_type in enumerate(kernel_types):
                for kp in kernel_params[kt]:
                    if virtual_label_type.endswith('_s'):
                        dv_file = os.path.join(dv_dir, f"dv_C={C}_{kernel_type}_{kp}.mat")
                    else:
                        dv_file = os.path.join(dv_dir, f"dv_round={r}_N={N}_C={C}_{kernel_type}_{kp}.mat")

                    decision_values = loadmat(dv_file)['decision_values']
                    all_test_dv.append(decision_values)

                    mmd_file = os.path.join(mmd_dir, f"mmd_{kernel_type}_{kp}.mat")
                    mmd_value = loadmat(mmd_file)['mmd_value']
                    mmd_values.append(mmd_value)

        reshaped_dv = [dv.reshape(-1) for dv in all_test_dv]
        f_s = np.column_stack(reshaped_dv)
        mmd_values = np.array(mmd_values)
        

        pp = -beta * (mmd_values ** 2)
        gamma_s = np.exp(pp)
        gamma_s = gamma_s / np.sum(gamma_s)
        gamma_s = np.array(gamma_s).flatten()
        

        virtual_label = np.dot(f_s, gamma_s)
        

        tilde_y = y_all.copy()
        tilde_y[src_index] = 0
        tilde_y[tar_test_index] = virtual_label[tar_test_index]
        

        add_kernel = np.ones(len(tilde_y))
        add_kernel[src_index] = 1.0 / lambda_D2
        add_kernel[tar_train_index] = 1.0 / lambda_L
        add_kernel[tar_test_index] = 1.0 / lambda_D1 / np.sum(gamma_s)
        

        cond = np.abs(virtual_label[tar_test_index]) < thr
        ind = np.where(cond)[0]
        tar_test_exclude = np.setdiff1d(tar_test_index, tar_test_index[ind])
        

        v_ind = np.concatenate((src_index, tar_train_index, tar_test_exclude))
        
        epsilon = 0.1
        
        K_v = K[np.ix_(v_ind, v_ind)] + np.diag(add_kernel[v_ind])
        
        temp_file = os.path.join(temp_dir, f"svm_data_round{r}.mat")
        print(f"Saving data to {temp_file}")
        savemat(temp_file, {
            'tilde_y': tilde_y[v_ind].reshape(-1, 1), 
            'v_ind': v_ind + 1, 
            'K_v': K_v,
            'K': K,
            'tar_index': tar_index + 1, 
            'C': C,
            'epsilon': epsilon
        })
        
        print(f"Round {r+1}/{data['nRound']}: Calling MATLAB for SVM training...")
        matlab_path = "F:\\MATLAB\\bin\\win64\\MATLAB.exe"  # Example:
        temp_file_unix = temp_file.replace("\\", "/")
        bat_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_matlab.bat")
       
        if not os.path.exists(bat_file):
            with open(bat_file, 'w') as f:
                f.write('@echo off\n')
                f.write('set MATLAB_PATH=F:\\MATLAB\\bin\\win64\\MATLAB.exe\n')
                f.write('set LIBSVM_PATH=F:/MATLAB/libsvm-3.35/libsvm-3.35/matlab\n')
                f.write('set DATA_FILE=%1\n')
                f.write('echo addpath(\'%%LIBSVM_PATH%%\'); > temp_script.m\n')
                f.write('echo disp(\'LIBSVM path added\'); >> temp_script.m\n')
                f.write('echo train_svm_model(\'%%DATA_FILE%%\'); >> temp_script.m\n')
                f.write('"%MATLAB_PATH%" -batch "run(\'temp_script.m\')"')

        try:
            process = subprocess.run([bat_file, temp_file_unix], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("MATLAB output:", process.stdout.decode('utf-8', errors='ignore'))
        except subprocess.CalledProcessError as e:
            print("MATLAB execution failed, return code:", e.returncode)
            print("MATLAB standard output:", e.stdout.decode('utf-8', errors='ignore'))
            print("MATLAB error output:", e.stderr.decode('utf-8', errors='ignore'))
            raise RuntimeError("MATLAB execution failed")
                
        result_data = loadmat(temp_file)
        
        if 'dv' not in result_data:
            raise RuntimeError("MATLAB training failed: Decision values not found")
            
        dv = result_data['dv'].flatten()
        print(f"Decision values shape: {dv.shape}")
        
        ap = calc_ap(data['yt'][tar_test_index], dv[tar_test_index-1])
        acc = np.mean(data['yt'][tar_test_index] == np.sign(dv[tar_test_index-1]))
        print(f"MATLAB training result: AP={ap:.4f}, Accuracy={acc:.4f}")
        
        result[r] = {'ap': ap, 'acc': acc}
        log_print(result_file,
                  "******%g\t%g @ round=%d, C=%g, lambda_L=%g, lambda_D1=%g, lambda_D2=%g, thr=%g\n",
                  ap, acc, r, C, lambda_L, lambda_D1, lambda_D2, thr)
    
    log_print(result_file,
              "<==========  END @ %s, C = %g, lambda_L = %g, lambda_D1 = %g, lambda_D2 = %g, thr = %g, beta = %g ===========>\n",
              np.datetime64('now'), C, lambda_L, lambda_D1, lambda_D2, thr, beta)
    
    return result
