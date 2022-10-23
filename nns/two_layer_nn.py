# Imports 
from os import access
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from customData import train_loader, test_loader, val_loader

# Create Fully connected network 
class NN(nn.Module): 
    def __init__(self, input_size, num_classes): #(224 X 224 = 50176 nodes)
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    
    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# Set Device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters 
input_size = 50176
num_classes = 3 
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Initialize Network 
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the network 
for epoch in range(num_epochs): 
    for batch_idx, (data, targets) in enumerate(train_loader): 
        # Get data to CUDA is possible 
        data = data.to(device = device)
        targets = targets.to(device = device)
        
        # Get to the correct shape 
        data = data.reshape(data.shape[0], -1)
        
        # forward 
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backward 
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model 
def check_accuracy(loader, model): 
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad(): 
        for x, y in loader: 
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum() # predictions with the correct label 
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()
    

        
print("Checking accuracy on training data")        
check_accuracy(train_loader, model)
print("Checking accuracy on test data")
check_accuracy(test_loader, model)
print("Checking accuracy on validation data")
check_accuracy(val_loader, model)

        
            