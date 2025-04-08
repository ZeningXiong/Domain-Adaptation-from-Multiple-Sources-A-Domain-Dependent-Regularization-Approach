import os
from scipy.io import savemat
from load_data import load_data
from main_fast_dam import main_fast_dam
from show_all_fast_dam import show_result_all_fast_dam

def run_fast_dam():
    
    print(f'============= {os.path.basename(__file__)} =================\n')

    
    data = load_data()

    
    C = 1
    lambda_L = 1
    lambda_D = 1
    beta = 100
    thr = 0
    virtual_label_type = 'svm_fr'
    kernel_types = ['linear', 'poly']
    kernel_params = [
        [0],  
        [1.1, 1.2, 1.3, 1.4, 1.5] 
    ]

    
    result = main_fast_dam(data, C, lambda_L, lambda_D, thr, beta, virtual_label_type, kernel_types, kernel_params)

   
    result_dir = 'results'
    
    result_file = os.path.join(result_dir, 'fast_dam', f'result_main_fast_dam_{virtual_label_type}.txt')
    
    with open(result_file, 'w') as f:
        f.write(f"result: {result}\n")
        f.write(f"C: {C}\n")
        f.write(f"lambda_L: {lambda_L}\n")
        f.write(f"lambda_D: {lambda_D}\n")
        f.write(f"beta: {beta}\n")
        f.write(f"kernel_types: {kernel_types}\n")
        f.write(f"kernel_params: {kernel_params}\n")
        f.write(f"thr: {thr}\n")
    
    show_result_all_fast_dam(virtual_label_type)


run_fast_dam()