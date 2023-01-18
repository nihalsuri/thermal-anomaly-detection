import torch
import timm
from PIL import Image
from torchvision import transforms, models
import cv2
import matplotlib.pyplot as plt
import time
import os 

def preprocess_tensor(path):
    img = cv2.imread(path)
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX)
    img = Image.fromarray(img)
    img = transforms.functional.crop(img, 190, 380, 140, 230)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    return img


def preprocess_img(path):
    img = cv2.imread(path)
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX)
    img = Image.fromarray(img)
    return img
    
    
def loader(folder_dir, input_test): 
    for images in os.listdir(folder_dir):
      if(images.endswith('.png')): 
        input = preprocess_tensor(folder_dir + '/' + images)
        input = torch.unsqueeze(input, 0)
        input = input.to('cpu')
        input_test.append(input)
        #return input_test


class_mapping = ['healthy', 'misalignment', 'rotor damage']
path_res = 'resnet10t.pt'


device = 'cpu'
resnet10t = timm.create_model('resnet10t', pretrained=True, num_classes=3)
model_resnet = resnet10t
model_resnet.load_state_dict(torch.load(path_res, map_location=torch.device('cpu')))


rotor_correct = 0
healthy_correct = 0
misalignment_correct = 0

healthy_tensors = []
rotor_tensors = []
misalignment_tensors = []

tik_total = time.time()
loader("data/healthy", healthy_tensors)
loader("data/rotor", rotor_tensors)
loader("data/misalignment", misalignment_tensors)

tik_inferencing = time.time()
model_resnet.eval()
with torch.no_grad():
  for i in range(len(rotor_tensors)):
      prediction = model_resnet(rotor_tensors[i])
      prediction_index = prediction[0].argmax(0)
      predicted_class_rotor = class_mapping[prediction_index]
      if predicted_class_rotor == 'rotor damage':
          rotor_correct+=1
      #print(predicted_class_rotor)

  for i in range(len(healthy_tensors)):
      prediction = model_resnet(healthy_tensors[i])
      prediction_index = prediction[0].argmax(0)
      predicted_class_healthy = class_mapping[prediction_index]
      if predicted_class_healthy == 'healthy':
          healthy_correct+=1
      #print(prediction_healthy)

  for i in range(len(misalignment_tensors)):
      prediction = model_resnet(misalignment_tensors[i])
      prediction_index = prediction[0].argmax(0)
      predicted_class_mis = class_mapping[prediction_index]
      if predicted_class_mis == 'misalignment':
          misalignment_correct+=1
      #print(predicted_class_mis)
  tok_inferencing = time.time()
  tok_total = time.time()

print(f'Accuracy for healthy class: [{healthy_correct}/40] - {healthy_correct / 40 * 100}%')
print(f'Accuracy for rotor damage class: [{rotor_correct}/40] - {rotor_correct / 40 * 100}%')
print(f'Accuracy for misalignment class: [{misalignment_correct}/40] - {misalignment_correct / 40 * 100}%')
print()
print('Total inferencing time: ', round(tok_inferencing - tik_inferencing, 4), '[s]')
print('Total preprocessing and inferencing time: ', round(tok_total - tik_total, 4), '[s]')
print()
print(f'Average inferencing time per image: {round((tok_inferencing - tik_inferencing)/120, 4)}[s]')
print(f'FPS: {1/((tok_inferencing - tik_inferencing)/120)}')
