from load_comp_vs_rec import load_comp_vs_rec
from load_rec_vs_sci import load_rec_vs_sci
from load_sci_vs_comp import load_sci_vs_comp
def load_data(setting):
   
   
    load_func_name = f'load_{setting}'
    
    
    load_func = globals().get(load_func_name)
    
    if not load_func:
        raise ValueError(f"Function {load_func_name} not found. Please define it first.")
    
    
    data = load_func()
    
    
    
    data['setting'] = setting
    
    
    return data
