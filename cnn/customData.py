from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import cv2


# Custom dataset class : returns image and label for futher training 
class ClutchDataset(Dataset):
    
    def __init__(self, dataframe, root_dir, is_train, transform=None): 
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self): 
        return len(self.dataframe)
    
    def __getitem__(self, idx): 
        if torch.is_tensor(idx): 
            idx = idx.tolist()
            
        img_name =  os.path.join(self.root_dir, self.dataframe.iloc[idx, 1], self.dataframe.iloc[idx, 0])
        image1 = cv2.imread(img_name)
        image = Image.fromarray(image1)
        
        if self.is_train: 
            labelKey = self.dataframe.iloc[idx, 1]
            label = torch.tensor(int(ids[labelKey]))
        else: 
            label = torch.tensor(1)
    
        if self.transform: 
            image = self.transform(image)
            
        return image, label


    
def createDataframe(path, name):
    
    cols = ["image_id", "label"]
    labelsCsv = open(name, 'w')
    writer = csv.writer(labelsCsv)
    writer.writerow(cols)
    
    dir = os.listdir(path)
    for subdir in dir:
        subdirContents = os.listdir(os.path.join(path, subdir))
        for contents in subdirContents: 
            col = [contents, subdir]
            writer.writerow(col)
    
    labelsCsv.close()


def removeEmptyRows(inputFile, outputFile):
    df = pd.read_csv(inputFile)
    df.to_csv(outputFile, index = False)



def label2id(labelArray):
    label2id = {}
    id2label = {}
    index = 0

    for class_name in labelArray: 
        label2id[class_name] = str(index)
        id2label[str(index)] = class_name
        index = index + 1
    
    return label2id
    


# dataframe created and csv file added 
path = "C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/clutch_segregated"
createDataframe(path, 'train.csv')
removeEmptyRows('train.csv', 'train_data.csv')

# label creations 
train_data = pd.read_csv('train_data.csv')
labelArr = train_data['label'].unique()
ids = label2id(labelArr)

# train test split 
train_data_sample = train_data.sample(frac = 0.4)
train, valid = train_test_split(train_data_sample, stratify=train_data_sample['label'], test_size=0.2)
# print(len(train))
# print(len(valid))


#Add transforms for train and test dataset 
input_size = 224
transform_train = transforms.Compose([
    # add other transformations in this list
    #transforms.Resize(input_size), 
    transforms.Grayscale(num_output_channels = 1), 
    transforms.ToTensor()
])

transform_valid = transforms.Compose([
    # add other transformations in this list
    #transforms.Resize(input_size), 
    transforms.Grayscale(num_output_channels = 1), 
    transforms.ToTensor()
])



# dataloaders
train_dataset = ClutchDataset(train, root_dir = path, is_train = True, transform=transform_train)
valid_dataset = ClutchDataset(valid, root_dir = path, is_train = False, transform=transform_valid)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(valid_dataset, batch_size = 64, shuffle = True)
dataloaders_dict = {}
dataloaders_dict['train'] = train_loader 
dataloaders_dict['val'] = val_loader


# iteratre through dataloaders 
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")






