import os
from scipy.io import savemat
from load_data import load_data
from main_univerdam_m import main_univerdam
from show_result_all_univer_dam import show_result_univer_dam

def run_univer_dam():
    
    print(f'============= {os.path.basename(__file__)} =================\n')

    data = load_data()

    C = 1
    lambda_L = 1
    lambda_D1 = 1
    lambda_D2 = 1
    thr = 0
    beta = 100
    virtual_label_type = 'svm_fr'
    kernel_types = ['linear', 'poly']
    kernel_params = [
        [0],
        [1.1, 1.2, 1.3, 1.4, 1.5]
    ]

    result = main_univerdam(data, C, lambda_L, lambda_D1, lambda_D2, thr, beta, 
                         virtual_label_type, kernel_types, kernel_params)

    result_dir = 'results'
    os.makedirs(os.path.join(result_dir, 'univerdam'), exist_ok=True)
    
    result_file = os.path.join(result_dir, 'univerdam', f'result_main_univerdam_m_{virtual_label_type}.txt')
    
    with open(result_file, 'w') as f:
        f.write(f"result: {result}\n")
        f.write(f"C: {C}\n")
        f.write(f"lambda_L: {lambda_L}\n")
        f.write(f"lambda_D1: {lambda_D1}\n")
        f.write(f"lambda_D2: {lambda_D2}\n")
        f.write(f"beta: {beta}\n")
        f.write(f"kernel_types: {kernel_types}\n")
        f.write(f"kernel_params: {kernel_params}\n")
        f.write(f"thr: {thr}\n")
    
    show_result_univer_dam(virtual_label_type)


run_univer_dam()