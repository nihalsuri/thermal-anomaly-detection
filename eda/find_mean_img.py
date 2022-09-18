import matplotlib.pyplot as plt
import numpy as np

def fmi(full_mat, title, size = (64, 64)):
    # calculate the average
    mean_img = np.mean(full_mat, axis = 0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(size)
    # plt.imshow(mean_img, vmin=0, vmax=255, cmap='Greys_r')
    # plt.title(f'Mean {title}')
    # plt.axis('off')
    # plt.show()
    return mean_img

