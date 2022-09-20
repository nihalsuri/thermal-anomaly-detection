import os 
import numpy as np 
import matplotlib.pyplot as plt
from keras_preprocessing import image
from images_as_matrix import img2np
from find_mean_img import mean_sd

#image folder 
segregated_dir = "C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/clutch_segregated"

#the list of images in the subfolders 
rotor_imgs = [fn for fn in os.listdir(f'{segregated_dir}/rotor') if fn.endswith('.png')]
misalignment_imgs = [fn for fn in os.listdir(f'{segregated_dir}/misalignment') if fn.endswith('.png')]
standard_imgs = [fn for fn in os.listdir(f'{segregated_dir}/standard') if fn.endswith('.png')]

#randomly select three images 
select_rotor = np.random.choice(rotor_imgs, 3, replace = False)
select_misalignment = np.random.choice(misalignment_imgs, 3, replace = False)
select_standard = np.random.choice(standard_imgs, 3, replace = False)

#plotting 3 x 3 image matrix 
fig = plt.figure(figsize = (8,6))

for i in range(9): 
    if i < 3: 
        fp = f'{segregated_dir}/rotor/{select_rotor[i]}'
        label = 'Rotor'
    elif i >= 3 and i < 6: 
        fp = f'{segregated_dir}/misalignment/{select_misalignment[i - 3]}'
        label = 'Misalignment'
    else: 
        fp = f'{segregated_dir}/standard/{select_standard[i - 6]}'
        label = 'Standard'
    
    ax = fig.add_subplot(3, 3, i+1)
    
    #to plot without the rescalling remove target_size
    fn = image.load_img(fp, color_mode= 'grayscale')
    plt.imshow(fn, cmap = 'Greys_r')
    plt.title(label)
    plt.axis('off')
# plt.show()


#run it on our folders 
rotor_images = img2np(f'{segregated_dir}/rotor', rotor_imgs)
misalignement_images = img2np(f'{segregated_dir}/misalignment', misalignment_imgs)
standard_images = img2np(f'{segregated_dir}/standard', standard_imgs)

#mean and sd 
rotor_mean, rotor_sd = mean_sd(rotor_images)
misalignment_mean, misalignment_sd = mean_sd(misalignement_images)
standard_mean, standard_sd = mean_sd(standard_images)

mean_images = [rotor_mean, misalignment_mean, standard_mean]
sd_images = [rotor_sd, misalignment_sd, standard_sd]
titles = ["ROTOR", "MISALIGNMENT", "STANDARD"]

fig_mean = plt.figure(figsize = (8,6))

for i in range(6): 
    mean_ax = fig_mean.add_subplot(2, 3, i+1) 
    
    if i < 3: 
        plt.imshow(mean_images[i], vmin=0, vmax=255, cmap='Greys_r')
        plt.title(f'Mean {titles[i]}')
        
    else: 
        plt.imshow(sd_images[i - 3], vmin=0, vmax=255, cmap='Greys_r')
        plt.title(f'Standard Deviation {titles[i - 3]}')
    plt.axis('off')

#compute the contrast between Rotor vs Misalignment and Standard vs Misalignment 
diffMean = plt.figure(figsize = (10, 10))
rotorVsmis = misalignment_mean - rotor_mean 
standardVsmis = misalignment_mean - standard_mean   

diffMean.add_subplot(1, 2, 1)
plt.imshow(rotorVsmis, cmap = 'bwr')
plt.title(f'Rotor vs Misalignment Mean Image')
plt.axis('off')

diffMean.add_subplot(1, 2, 2)
plt.imshow(standardVsmis, cmap = 'bwr')
plt.title(f'Standard vs Misalignment Mean Image')
plt.axis('off')






plt.show()



    




