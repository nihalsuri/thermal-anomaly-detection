from processing import normDir
from shutil import copy
from glob import iglob 
import os

newDir = "C:/Users/nihal.suri/Desktop/Courses/Thesis/clutch_segregated"

rotorDir = os.path.join(newDir, "rotor")
misalignmentDir = os.path.join(newDir, "misalignment")
standardDir = os.path.join(newDir, "standard")

if os.path.exists(normDir):
    for subdir in os.listdir(normDir): 
        d = os.path.join(normDir, subdir)
        if os.path.isdir(d):
            if not os.path.exists(newDir): 
                os.mkdir(newDir)
                os.mkdir(rotorDir)
                os.mkdir(misalignmentDir) 
                os.mkdir(standardDir)
            
            if "rotor" in subdir:
                #print(d)
                for pngfile in iglob(os.path.join(d, "*.png")): 
                    copy(pngfile, rotorDir)  
                
                
            elif "misalignment" in subdir:
                #print(d)
                for pngfile in iglob(os.path.join(d, "*.png")): 
                    copy(pngfile, misalignmentDir)               
                
                
            elif "rotor" not in subdir and "misalignment" not in subdir:
                #print(d)
                for pngfile in iglob(os.path.join(d, "*.png")): 
                    copy(pngfile, standardDir)            
                
            
            
                
                
            
    
            
            
            