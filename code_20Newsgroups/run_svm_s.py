import os
import numpy as np
from scipy.io import savemat
from load_data import load_data
from main_svm_s import main_svm_s
from show_result_all_svm_s import show_result_all_svm_s
from load_comp_vs_rec import load_comp_vs_rec
from load_rec_vs_sci import load_rec_vs_sci
from load_sci_vs_comp import load_sci_vs_comp
from scipy.io import loadmat

def run_svm_s(setting):
    
    
    print(f'=========== {os.path.basename(__file__)} ============\n')
    
    
    data = load_data(setting)
    
    
    result_dir = f'result_{setting}'
    os.makedirs(result_dir, exist_ok=True)
    
   
    C = 1
    func = main_svm_s  
    
    
    kernel_types = ['linear', 'poly']
    kernel_params = [
        [0],  
        [1.1, 1.2, 1.3, 1.4, 1.5] #  1.1,1.2,...1.5
    ]
    N_set = [0, 2, 4, 6, 10, 15, 20]
    
    for N in N_set:
        
        result = func(data, C, N, kernel_types, kernel_params)
        
        
        result_file = os.path.join(result_dir, f'result_{func.__name__}_N={N}.txt')
        with open(result_file, 'w') as f:
            f.write(f"result: {result}\n")
            f.write(f"C: {C}\n")
            f.write(f"kernel_types: {kernel_types}\n")
            f.write(f"kernel_params: {kernel_params}\n")
        
        
    
    
    show_result_all_svm_s(setting, func)
    

s = 'rec_vs_sci'
run_svm_s(s)
    
