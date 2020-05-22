import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


batch_size = 1
lr = 1e-4
num_workers = 8
num_epochs = 100

# Image transformations
image_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]) 
    ])

# Load data
train_dataset = ImageFolder(
    root= 'data',
    transform=image_transforms
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True
)

# Define model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 15, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc1 = nn.Linear(2535, 512)          
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

model = ConvNet()

# Send model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function
criterion = nn.L1Loss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train model
print('Training model...')

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
        
    model.train()

    running_loss = 0.0
    successes = 0.0
    total = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Set gradients to zero
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)

            # Counting L1 loss
            loss =  criterion(outputs, labels.float())
            
            # Success metric
            l_outputs= outputs.round().flatten().tolist()
            l_labels=labels.float().tolist()

            for x in range(0, len(l_labels)):
                if l_outputs[x] == l_labels[x]:
                    successes += 1
            total+= len(l_labels)

            # Backpropagate
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)

    successes = successes / total
    epoch_loss = running_loss / (len(train_loader) * batch_size)
    print('Loss: {:.4f} Successes: {:.4f}'.format(
        epoch_loss, successes))

    print()

