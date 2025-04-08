import numpy as np
from sklearn.svm import SVC
import re

def svmtrain(y, X, options):
    """
    A Python wrapper that mimics the MATLAB LIBSVM svmtrain interface.
    
    Inputs:
        y       : Training label vector (numpy array of shape (m,))
        X       : Training instance matrix (numpy array of shape (m, n)). 
                  For precomputed kernel, the first column should be sample indices.
        options : A string of training options in LIBSVM format (e.g., "-s 3 -c 1 -t 4 -q -p 0.1")
    
    Outputs:
        model_dict : A dictionary containing model parameters (mimicking LIBSVM's structure)
        model      : The trained scikit-learn SVC model.
    
    Note:
        - Currently, this wrapper mainly supports precomputed kernels (-t 4) and a linear kernel.
        - It parses the option string to extract the regularization parameter (-c),
          kernel type (-t), epsilon parameter (-p) for regression (if needed) and probability flag (-b).
        - For precomputed kernels, it assumes that the first column of X contains sample indices,
          which will be removed before training.
    """
    
    
    match_c = re.search(r'-c\s*([\d\.]+)', options)
    C = float(match_c.group(1)) if match_c else 1.0
    
    
    match_t = re.search(r'-t\s*(\d+)', options)
    t = int(match_t.group(1)) if match_t else 0  # default kernel: linear (0)
    
    
    match_p = re.search(r'-p\s*([\d\.]+)', options)
    epsilon = float(match_p.group(1)) if match_p else 0.1
    
    
    match_b = re.search(r'-b\s*(\d+)', options)
    probability = bool(int(match_b.group(1))) if match_b else False
    
    
    if t == 4:
        X_kernel = X[:, 1:]
        model = SVC(C=C, kernel='precomputed', probability=probability)
        model.fit(X_kernel, y)
    else:
        
        model = SVC(C=C, kernel='linear', probability=probability)
        model.fit(X, y)
    
    
    model_dict = {
        'Parameters': {
            'C': C,
            'kernel': 'precomputed' if t == 4 else 'linear',
            'epsilon': epsilon,
            'probability': probability
        },
        'nr_class': len(np.unique(y)),
        'totalSV': len(model.support_),
        'rho': -model.intercept_[0] if model.intercept_.ndim == 1 else -model.intercept_,
        'Label': np.unique(y),
       
        'nSV': None,
        'sv_coef': model.dual_coef_,
        'SVs': model.support_
    }
    
    return model_dict, model
