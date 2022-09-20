import matplotlib.pyplot as plt
import numpy as np

def mean_sd(full_mat, size = (64, 64)):
    # calculate the average
    mean_img = np.mean(full_mat, axis = 0, dtype=np.float64)
    sd_img = np.std(full_mat, axis = 0, dtype=np.float64 )
    # reshape it back to a matrix
    mean_img = mean_img.reshape(size)
    sd_img = sd_img.reshape(size)
    return mean_img, sd_img

