import numpy as np

def return_ind2(tar_training_ind, tar_pos_len, nSample):
    """
    Python implementation of the MATLAB `return_ind2` function.

    Inputs:
        tar_training_ind : Array of indices representing the target training data.
        tar_pos_len : Number of positive samples in the target training data.
        nSample : Number of samples to select from both positive and negative classes.

    Outputs:
        tar_training_ind : Indices of the selected training samples.
        test_ind : Indices of the remaining samples (test set).
    """
    if len(tar_training_ind) < tar_pos_len:
        raise ValueError("tar_training_ind must have at least tar_pos_len elements.")

    # Select the first nSample positive and nSample negative samples
    pos_indices = tar_training_ind[:tar_pos_len]  # Positive class indices
    neg_indices = tar_training_ind[tar_pos_len:]  # Negative class indices

    # Ensure nSample does not exceed the available samples
    n_pos = min(nSample, len(pos_indices))
    n_neg = min(nSample, len(neg_indices))

    # Select the first nSample from positive and negative classes
    selected_pos = pos_indices[:n_pos]
    selected_neg = neg_indices[:n_neg]

    # Combine selected indices to form the training set
    tar_training_ind1 = np.concatenate([selected_pos, selected_neg])

    # The remaining indices form the test set
    test_ind = np.setdiff1d(tar_training_ind, tar_training_ind1)

    tar_training_ind = tar_training_ind1[:]

    return tar_training_ind, test_ind


