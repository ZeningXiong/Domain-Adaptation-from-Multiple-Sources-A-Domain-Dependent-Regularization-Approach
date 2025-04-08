import os
import numpy as np
from load_data import load_data
from show_result_all_univer_dam import show_result_all_univer_dam
from main_univerdam_m import main_univerdam_m

def run_univer_dam(setting):

    print(f'============== {__name__} ===============\n')

    data = load_data(setting)

    result_dir = f'result_{setting}'
    os.makedirs(result_dir, exist_ok=True)

    C = 1
    lambda_L = 1
    lambda_D1 = 1
    lambda_D2 = 1
    beta = 10000
    thr = 0.3
    virtual_label_type = 'svm_s'
    kernel_types = ['linear', 'poly']

    kernel_params = [[0], [1.1, 1.2, 1.3, 1.4, 1.5]]

    N_set = [0, 2, 4, 6, 10, 15, 20]

    for N in N_set:
        result = main_univerdam_m(
            data, 
            C, 
            N, 
            lambda_L, 
            lambda_D1, 
            lambda_D2, 
            thr, 
            beta, 
            virtual_label_type, 
            kernel_types, 
            kernel_params
        )

        result_file = os.path.join(result_dir, f'result_main_univer_dam_{virtual_label_type}_N={N}.txt')
        with open(result_file, 'w') as f:
            f.write(f"result: {result}\n")
            f.write(f"C: {C}\n")
            f.write(f"lambda_L: {lambda_L}\n")
            f.write(f"lambda_D1: {lambda_D1}\n")
            f.write(f"lambda_D2: {lambda_D2}\n")
            f.write(f"beta: {beta}\n")
            f.write(f"thr: {thr}\n")
            f.write(f"kernel_types: {kernel_types}\n")
            f.write(f"kernel_params: {kernel_params}\n")


    show_result_all_univer_dam(setting, virtual_label_type)


if __name__ == "__main__":
    # setting = 'comp_vs_rec' 
    # setting = 'rec_vs_sci'
    setting = 'sci_vs_comp'
    run_univer_dam(setting)
