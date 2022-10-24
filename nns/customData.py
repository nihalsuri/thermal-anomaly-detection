from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2


# Data path with train, test and val segregation
data_path = 'C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/nns/dataset_distribution.csv'
df = pd.read_csv(data_path)

# Split data into three seperate dataframes : train , test , val
train_df = df[df['dataset'].str.contains('train')]
test_df = df[df['dataset'].str.contains('test')]
val_df = df[df['dataset'].str.contains('val')]


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

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])
        image1 = cv2.imread(img_name)
        image_norm = cv2.normalize(
            image1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image = Image.fromarray(image_norm)
        label = torch.tensor(int(self.dataframe.iloc[idx, 2]))

        if self.transform:
            image = self.transform(image)

        return image, label


# main dataset path
path = "C:/Users/nihal.suri/Documents/GitHub/thermal-anomaly-detection/clutch_2"


# Add transforms for train, test, val dataset
input_size = 224
transform_train = transforms.Compose([
    # add other transformations in this list
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.Resize((input_size, input_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()

    # transforms.RandomHorizontalFlip(p=0.5)
])

transform_valid = transforms.Compose([
    # add other transformations in this list
    transforms.Resize((input_size, input_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    # add other transformations in this list
    transforms.Resize((input_size, input_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


# dataloaders
train_dataset = ClutchDataset(
    train_df, root_dir=path, is_train=True, transform=transform_train)
val_dataset = ClutchDataset(val_df, root_dir=path,
                            is_train=False, transform=transform_valid)
test_dataset = ClutchDataset(
    test_df, root_dir=path, is_train=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

dataloaders_dict = {}
dataloaders_dict['train'] = train_loader
dataloaders_dict['val'] = val_loader
dataloaders_dict['test'] = test_loader


# iteratre through dataloaders
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
