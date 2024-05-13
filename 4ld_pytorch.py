import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import random

import torch.optim as optim

# Function that shows 5 original images and their corresponding 5 augmented images
def show_original_and_augmented(train_dataset, num_images=5):
    indices = random.sample(range(len(train_dataset)), num_images)

    original_images = [train_dataset[i][0] for i in indices]
    augmented_images = [train_dataset[i][0] for i in indices]

    fig, axes = plt.subplots(2, num_images, figsize=(10, 4))

    for i, img in enumerate(original_images):
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

    for i, img in enumerate(augmented_images):
        axes[1, i].imshow(img.squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Augmented")

    plt.savefig("original vs augmented images PYTORCH.png")
    plt.show()

# Function that shows best models prediction results of 5 images
def show_images_with_predictions(model_state, data_loader, num_images=5):
    # Load the model
    model = SDNT()
    model.load_state_dict(model_state)
    model.eval()

    # Get some images and their labels from the data loader
    images, labels = next(iter(data_loader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)

    # Plot the images with labels and predictions
    fig, axes = plt.subplots(1, num_images, figsize=(12, 5))
    for i in range(num_images):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}\nPrediction: {predicted[i]}\nProbability: {probabilities[i][predicted[i]]:.2f}")
        axes[i].axis('off')

    plt.savefig("best model predictions PYTORCH.png")
    plt.show()

# Load MNIST dataset: Load the MNIST dataset and apply transformations.
# Define image augmentation transforms
augmentation_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Random rotation within ±15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random horizontal and vertical shift within ±10%
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # Random zoom within ±10%
    transforms.RandomHorizontalFlip(),  # Random horizontal flipping
    transforms.RandomVerticalFlip(),  # Random vertical flipping
    transforms.ColorJitter(brightness=0.2),  # Random brightness adjustment within ±0.2
    transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),  # Random crop and resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=augmentation_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define SDNT model: Define the architecture of the SDNT model. We'll create a simple model with fully connected layers.
class SDNT(nn.Module):
    def __init__(self):
        super(SDNT, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding=1) # 128 YRA OUTPUTAS

        self.dropout = nn.Dropout2d(p=0.25)
        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust input size after pooling
        self.fc2 = nn.Linear(128, 10)  # Output size is 10 (number of classes)

    def forward(self, x):
        # Convolutional layers with relu activation and pooling
        # EACH LAYER OF CONVOLUSION CUTS THE RESOLUTION BY 2 !!!!!!!!! 28/2 = 14; 14/2 = 7; 7/2 = 4
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)

        # NEPAVYKO SITOS IMPLEMENTUOT:)
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.dropout(x)

        # Flatten the input
        x = x.view(-1, 64 * 7 * 7)  # Adjust to the output size after pooling
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Instantiate the model and define loss function and optimizer:
model = SDNT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop: Train the SDNT model.
def train(model, train_loader, criterion, optimizer, epochs, save_path):
    best_accuracy = 0.0
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        
        # Evaluate the model on validation dataset
        accuracy = test(model, test_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
        
    # Save the best model
    torch.save(best_model_state, save_path + "_best.pth")
    
    # Save the last model
    torch.save(model.state_dict(), save_path + "_last.pth")

    return best_model_state

# Testing loop: Evaluate the model on the test dataset.
def test(model, test_loader):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)
    return accuracy


# Displaying images
show_original_and_augmented(train_dataset)

# Train the model
best_trained_model = train(model, train_loader, criterion, optimizer, epochs=1, save_path="pytorch_files\mnist_model")

show_images_with_predictions(best_trained_model, test_loader)

# Test the model
test(model, test_loader)
