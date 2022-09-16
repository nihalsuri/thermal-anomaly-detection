import cv2
import os


rootdir = "C:/Users/nihal.suri/Desktop/Courses/Thesis/clutch_2"
normDir = rootdir + "_normalized"

if not os.path.exists(normDir):
    os.mkdir(normDir) 
    

    for rootdir, dirs, files in os.walk(rootdir): 

        for subdir in dirs: 
            currDir = rootdir + '/' + subdir
            subNormdir = normDir + '/' + subdir + '_normalized'
            
            if not os.path.exists(subNormdir): 
                os.mkdir(subNormdir)
        
                for images in os.listdir(currDir):
            
                    if(images.endswith(".png")):
                        imgPath = rootdir + '/' + subdir + '/' + images
                        #print(imgPath)
                        image = cv2.imread(imgPath)
                        image_norm = cv2.normalize(image, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX); 
                        normImg = 'norm' + images  
                        cv2.imwrite(subNormdir + '/' + normImg, image_norm)
                        print(images)
                
else : print("Images are already processed...")                
        



