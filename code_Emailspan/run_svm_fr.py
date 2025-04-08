import os
from scipy.io import savemat
from load_data import load_data
import numpy as np
from main_svm_fr import main_svm_fr
from show_all_svm_fr import show_result_all_svm_fr

def run_svm_fr():
    
    print(f'============= {os.path.basename(__file__)} =================\n')

    
    data = load_data()

   
    C = 1
    kernel_types = ['linear', 'poly']
    kernel_params = [
        [0],  
        [1.1, 1.2, 1.3, 1.4, 1.5]  
    ]

  
    result = main_svm_fr(data, C, kernel_types, kernel_params)

    
    kernel_types = ['linear']
    kernel_params = [[0]] 
    result = main_svm_fr(data, C, kernel_types, kernel_params)

   
    result_dir = 'results'
    
    result_file = os.path.join(result_dir, 'svm_fr', f'result_{os.path.basename(__file__)}.txt')
    
    with open(result_file, 'w') as f:
        f.write(f"result: {result}\n")
        f.write(f"C: {C}\n")
        f.write(f"kernel_types: {kernel_types}\n")
        f.write(f"kernel_params: {kernel_params}\n")
    show_result_all_svm_fr()

run_svm_fr()

