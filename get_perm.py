import numpy as np

def get_perm(num_data, portion):
    return np.random.permutation(num_data)[0: int(num_data * portion)]
