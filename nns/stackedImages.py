from PIL import Image
  
img_01 = Image.open("C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/cnn/transformation_results/standard/center_crop.png")
img_02 = Image.open("C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/cnn/transformation_results/standard/random_affine.png")
img_03 = Image.open("C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/cnn/transformation_results/standard/random_perspective.png")
img_04 = Image.open("C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/cnn/transformation_results/standard/random_rotation.png")
img_05 = Image.open("C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/cnn/transformation_results/standard/rand_resized_crop.png")


img_01_size = img_01.size
img_02_size = img_02.size
img_03_size = img_02.size
img_02_size = img_02.size

  
print('img 1 size: ', img_01_size)
print('img 2 size: ', img_02_size)
print('img 3 size: ', img_03_size)
print('img 4 size: ', img_03_size)
  
new_im = Image.new('RGB', (1*img_01_size[0],5*img_01_size[1]), (250,250,250))
  
new_im.paste(img_01, (0,0))
new_im.paste(img_02, (0, img_01_size[1]))
new_im.paste(img_03, (0, 2*img_01_size[1]))
new_im.paste(img_04, (0, 3*img_01_size[1]))
new_im.paste(img_05, (0, 4*img_01_size[1]))

new_im.save("transformation_results/standard/stacked.png", "PNG")
new_im.show()