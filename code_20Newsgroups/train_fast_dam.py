import numpy as np
from sklearn.svm import SVR

def train_fast_dam(K, y, f_s, gamma_s, theta1, theta2, thr, f_p=None, lambda_=1.0, C=1.0, epsilon=0.1):
    """
    Python implementation of the MATLAB `train_fast_dam` function.

    Inputs:
        K : n x n kernel matrix
        y : n x 1 label vector (labeled data: -1 or 1, unlabeled data: 0)
        f_s : virtual labels
        gamma_s : weights for virtual labels
        theta1, theta2, C : regularization parameters
        thr : threshold for filtering unlabeled data
        f_p : prior information (optional)
        lambda_ : regularization parameter for prior information
        epsilon : epsilon parameter for SVR

    Outputs:
        dv : decision values
        model : trained SVR model
    """
    
    assert np.all(np.isin(np.unique(y), [-1, 0, 1])), "Labels must be in {-1, 0, 1}."
   

    n = K.shape[0]
    idx_l = np.where(y != 0)[0]  
    idx_u = np.where(y == 0)[0]  

    
    II = np.ones(n)
    II[idx_u] = 1 / np.sum(gamma_s)
    II[idx_l] = II[idx_l] / theta1
    II[idx_u] = II[idx_u] / theta2
    II = np.diag(II)

    
    if f_p is not None:
        Kp = np.outer(f_p, f_p) / lambda_
        hatK = K + II + Kp
    else:
        hatK = K + II
        hatK = hatK.A

    
    
    
    haty = np.zeros(n)
    for i in range(len(gamma_s)):
        p = f_s[i]*gamma_s[i]
        haty += p[0]
    
    
    haty[idx_l] = y[idx_l]

    
    ind = np.where(np.abs(haty[idx_u]) < thr)[0]
    v_ind = np.concatenate([idx_l, np.setdiff1d(idx_u, idx_u[ind])])

    
    model = SVR(kernel="precomputed", C=C, epsilon=epsilon)
    model.fit(hatK[v_ind, :][:, v_ind], haty[v_ind])

    
    if f_p is not None:
        Ktest = K + Kp
    else:
        Ktest = K

    dv = Ktest[:, v_ind[model.support_]].dot(model.dual_coef_.T) - model.intercept_

    return dv, model


