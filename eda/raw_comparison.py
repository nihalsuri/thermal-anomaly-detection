import os 
import numpy as np 
import matplotlib.pyplot as plt
from keras_preprocessing import image

#image folder 
segregated_dir = "C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/clutch_segregated"

#the list of images in the subfolders 
rotor_imgs = [fn for fn in os.listdir(f'{segregated_dir}/rotor') if fn.endswith('.png')]
misalignemnt_imgs = [fn for fn in os.listdir(f'{segregated_dir}/misalignment') if fn.endswith('.png')]
standard_imgs = [fn for fn in os.listdir(f'{segregated_dir}/standard') if fn.endswith('.png')]

#randomly select three images 
select_rotor = np.random.choice(rotor_imgs, 3, replace = False)
select_misalignment = np.random.choice(misalignemnt_imgs, 3, replace = False)
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
plt.show()

#also check the number of files here 
print(len(rotor_imgs), len(misalignemnt_imgs), len(standard_imgs))

 
         




