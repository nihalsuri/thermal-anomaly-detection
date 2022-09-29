from sklearn.utils import shuffle
import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

labelArray = ["rotor", "misalignment", "standard"]

label2id = {}
id2label = {}
index = 0

for class_name in labelArray: 
    label2id[class_name] = str(index)
    id2label[str(index)] = class_name
    index = index + 1
print(label2id)