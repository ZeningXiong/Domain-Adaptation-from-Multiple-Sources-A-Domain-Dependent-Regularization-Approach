import os
import numpy as np
import ast

def show_result_all_univer_dam(setting, virtual_label_type):
    result_dir = f'result_{setting}'
    N_set = [0, 2, 4, 6, 10, 15, 20]

    for N in N_set:
        result_file = os.path.join(result_dir, f'result_main_univer_dam_{virtual_label_type}_N={N}.txt')
        
        data = {}
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    key, value = line.split(": ", 1)
                    data[key] = ast.literal_eval(value)
                    
        result = data['result']
        ap_values = []
        for ap in result.values():
            ap_values.append(ap['ap'])
        
        mean_ap = np.mean(ap_values)
        std_ap = np.std(ap_values, ddof=1)
        
        print(f"{mean_ap}\t{std_ap}")

    print("------")
    
    for N in N_set:
        result_file = os.path.join(result_dir, f'result_main_univer_dam_{virtual_label_type}_N={N}.txt')
        
        data = {}
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    key, value = line.split(": ", 1)
                    data[key] = ast.literal_eval(value)
                    
        result = data['result']
        ap_values = []
        for ap in result.values():
            ap_values.append(ap['ap'])
            
        mean_ap = np.mean(ap_values)
        
        print(f"{mean_ap}")
