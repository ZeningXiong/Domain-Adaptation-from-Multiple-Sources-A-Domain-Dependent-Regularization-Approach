import os
import numpy as np
from scipy.io import savemat
from load_comp_vs_rec import load_comp_vs_rec
from main_fast_dam import main_fast_dam
from show_result_all_fast_dam import show_result_all_fast_dam
from load_data import load_data
from load_rec_vs_sci import load_rec_vs_sci


def run_fast_dam(setting):
    
    
    print(f'============== {__name__} ===============\n')

    
    data = load_data(setting)

    
    result_dir = f'result_{setting}'
    os.makedirs(result_dir, exist_ok=True)

    
    C = 1
    lambda_L = 1
    lambda_D = 1
    beta = 10000
    thr = 0
    virtual_label_type = 'svm_s'
    kernel_types = ['linear', 'poly']
    kernel_params = [[0], [1.1, 1.2, 1.3, 1.4, 1.5]]  # 1.1:0.1:1.5

    
    N_set = [0, 2, 4, 6, 10, 15, 20]
    for N in N_set:
        
        result = main_fast_dam(
            data, C, N, lambda_L, lambda_D, thr, beta, 
            virtual_label_type, kernel_types, kernel_params
        )

        
        result_file = os.path.join(result_dir, f'result_main_fast_dam_{virtual_label_type}_N={N}.txt')
        with open(result_file, 'w') as f:
            f.write(f"result: {result}\n")
            f.write(f"C: {C}\n")
            f.write(f"kernel_types: {kernel_types}\n")
            f.write(f"kernel_params: {kernel_params}\n")

    show_result_all_fast_dam(setting, virtual_label_type)


setting = 'rec_vs_sci'
run_fast_dam(setting)