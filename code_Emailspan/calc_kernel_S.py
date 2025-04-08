def calc_kernel_S(kernel_type, param, K):
   
    if kernel_type == 'linear':
        return K
    elif kernel_type == 'poly':
        return (K + 1) ** param
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")