import torch
import numpy as np
import os
import json
import argparse
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image

# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# Define the transforms to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test dataset
test_dir = 'path/to/test/folder'
test_dataset = ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define a function to predict the class of an image
def predict_class(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to a tensor and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)

    # Move the tensor to the specified device
    image_tensor = image_tensor.to(device)

    # Pass the tensor through the model
    output = model(image_tensor)

    # Get the class with the highest probability
    _, predicted_class = torch.max(output, 1)

    # Return the class
    return predicted_class.item()

# Define a function to predict the class of an image and return the label
def predict_class_with_label(image_path):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to a tensor and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)

    # Move the tensor to the specified device
    image_tensor = image_tensor.to(device)

    # Pass the tensor through the model
    output = model(image_tensor)

    # Get the class with the highest probability and the corresponding label
    probs, predicted_class = torch.max(output, 1)
    label_path = os.path.join(test_dir, test_dataset.classes[predicted_class.item()], 'label.txt')
    with open(label_path, 'r') as f:
        label = json.load(f)
    return predicted_class.item(), label

# Predict the class of an image
image_path = 'path/to/image'
class_id = predict_class(image_path)
print(f'The image is classified as {class_id}')

# Predict the class of an image and return the label
class_id, label = predict_class_with_label(image_path)
print(f'The image is classified as {class_id} and the label is {label}')

if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Parser for a preddiction of faces')
    parser.add_argument('directory', type=str, default='inference/', help='Path to the image')

    # Parse the arguments
    args = parser.parse_args()

    test_directory = args.directory
    
