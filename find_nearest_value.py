import numpy as np

def find_nearest(array, value):
    # find the nearest value in an array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx