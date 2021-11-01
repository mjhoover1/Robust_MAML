# -*- coding: utf-8 -*-
"""A file to store all poisoning APIs"""

import torch
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *


# TODO: Is this supposed to make the support image less opaque?
def poison_image_watermark(support_img, target_img, opacity, show=False):
    """
    Recursively retrieves the autograd graph for a particular tensor.
    # Arguments
        support_img: Image from the support set.
        target_img: Specific img from the query set, the target img to be misclassified.
        opacity: Amount of opacity to retain for base image, so target opacity is 1-opacity.
        show: If true will show support, target, and poisoned image.
    # Returns
        poisoned_img: A poisoned support img with target watermark
    """ 
    # Testing with 50%
    poisoned_img = (1-opacity)*target_img + opacity*support_img # opacity of target image for overlay on base images
    if show:
      show_img(support_img, 'Support Img')
      show_img(target_img, 'Target Img')
      show_img(poisoned_img, 'Poisoned Img')
      
    return poisoned_img


# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(image, applied_patch, mask, target, probability_threshold, model, lr=1, max_iteration=100):
    # Don't know what this does
    model.eval()
    
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    
    # Apply the patch to the image using the mask
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    
    # While the target probability is less than the desired threshold and not iterating out...
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        
        # Optimize the patch
        # Creates a variable and its gradient among other things in perturbed_image
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        
        # Get the logits of perturbed image
        output = model(per_image)
        
        # Evaluate model on target
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        
        # Move backwards... why not just not use gradient at all above?
        target_log_softmax.backward()
        
        # Get the gradient of perturbed image... though I don't know with respect to what
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        
        # Modify the patch to be "more optimal"
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        
        # Reapply the new patch to the image
        # Test the patch and get logits of perturbed image
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch

# Initialize the patch
def patch_initialization(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
    if patch_type == 'rectangle':
        # Takes dimensions of image, using percentage of "noise" gives the total
        # area of the patch. Taking sqrt gives size of one side
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        
        # Makes a random square matrix with same amount of layers as image
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

# Generate the mask and apply the patch
def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 224, 224)):
    # Initialize blank image
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle':
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        
        # patch location
        x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
        for i in range(patch.shape[0]):
            # Apply patch onto "blank" image
            applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    
    # Create a mask for every pixel that is a patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    
    return applied_patch, mask, x_location, y_location

# Test the patch on dataset
def test_patch(patch_type, target, patch, test_loader, model):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
    return test_success / test_actual_total