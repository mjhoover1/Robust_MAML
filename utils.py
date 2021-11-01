# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 18:50:43 2021

@author: bob12
"""

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# For patch attack
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Shows a given image 
def show_img(img, title, fig_dim = [10, 10]):
    """
    Recursively retrieves the autograd graph for a particular tensor.
    # Arguments
        img: Given image to show.
        fig_dim: Dimensions of the figure for examples [10, 10].
        title: The title desired for the shown image.
    # Returns
        Void
    # Side effect
        Shows a given image
    """ 
    plt.figure(figsize=fig_dim)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    
# Shows a given image group on a grid 
def show_grid(img_group, title, row_size, debug = False, fig_dim = [10, 10]):
    """
    Recursively retrieves the autograd graph for a particular tensor.
    # Arguments
        img: Given image group to show.
        title: The title desired for the shown image.
        fig_dim: Dimensions of the figure for examples [10, 10].
    # Returns
        Void
    # Side effect
        Shows a given image
    """ 
    grid_img = make_grid(img_group.squeeze(), nrow=row_size) # make grid (row_size and as many columns as possible) to display our n images
    if debug:
      print('images.shape:', img_group.shape)
      print('grid shape: ', grid_img.shape)
    show_img(grid_img, title, fig_dim)
    
    
# For patch attacks
# Load the datasets
# We randomly sample some images from the dataset, because ImageNet itself is too large.
def dataloader(train_size, test_size, data_dir, batch_size, num_workers, total_num=50000):
    # Setup the transformation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    index = np.arange(total_num)
    np.random.shuffle(index)
    train_index = index[:train_size]
    test_index = index[train_size: (train_size + test_size)]

    train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index), num_workers=num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_index), num_workers=num_workers, pin_memory=True, shuffle=False)
    return train_loader, test_loader

# Test the model on clean dataset
def test(model, dataloader):
    model.eval()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for (images, labels) in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    return correct / total

# Load the log and generate the training line
def log_generation(log_dir):
    # Load the statistics in the log
    epochs, train_rate, test_rate = [], [], []
    with open(log_dir, 'r') as f:
        reader = csv.reader(f)
        flag = 0
        for i in reader:
            if flag == 0:
                flag += 1
                continue
            else:
                epochs.append(int(i[0]))
                train_rate.append(float(i[1]))
                test_rate.append(float(i[2]))

    # Generate the success line
    plt.figure(num=0)
    plt.plot(epochs, test_rate, label='test_success_rate', linewidth=2, color='r')
    plt.plot(epochs, train_rate, label='train_success_rate', linewidth=2, color='b')
    plt.xlabel("epoch")
    plt.ylabel("success rate")
    plt.xlim(-1, max(epochs) + 1)
    plt.ylim(0, 1.0)
    plt.title("patch attack success rate")
    plt.legend()
    plt.savefig("training_pictures/patch_attack_success_rate.png")
    plt.close(0)