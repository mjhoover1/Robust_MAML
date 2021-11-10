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
def patch_attack(image_set, applied_patch, mask, target, query_attack_idx, probability_threshold, model, fast_weights,lr=1, max_iteration=100):
    # Don't know what this does
    model.eval()
    idx_to_poison = query_attack_idx.item()
    query_attack_idx = query_attack_idx.item()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    
    target_probability, count = 0, 0
    image_set_poisoned = image_set
    max_probability = probability_threshold
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image_set[idx_to_poison].type(torch.FloatTensor))
    image_set_poisoned[idx_to_poison] = perturbated_image
    # While the target probability is less than the desired threshold and not iterating out...
    while target_probability < probability_threshold and count < max_iteration:
        count += 1

        # Optimize the patch
        # Creates a variable and its gradient among other things in perturbed_image
        perturbated_image_set = Variable(image_set_poisoned, requires_grad=True)


        # Get the logits of perturbed image
        logits = model(perturbated_image_set, fast_weights, bn_training=True)

        # Evaluate model on target
        # 75 images total
        # 1x75x5
#         print(logits)
        target_log_softmax = torch.nn.functional.log_softmax(logits, dim=1)#[0][target]
#         pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        # Move backwards... why not just not use gradient at all above?

#         print(target_log_softmax[query_attack_idx.item()][target])

        target_log_softmax = target_log_softmax[idx_to_poison][target]
#         print(target_log_softmax)
        target_log_softmax.backward()

        # Get the gradient of perturbed image... though I don't know with respect to what
        patch_grad = perturbated_image_set.grad[idx_to_poison].clone().cpu()
#         print(patch_grad)
        perturbated_image_set.grad.data.zero_()
#         print(patch_grad)
        # Modify the patch to be "more optimal"
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        
        applied_patch = torch.mul(applied_patch,mask)
        # Reapply the new patch to the image
        # Test the patch and get logits of perturbed image
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image_set[idx_to_poison].type(torch.FloatTensor))
        
        
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        # perturbated_image = perturbated_image.cuda()
#         print(perturbated_image.size())
#         print(perturbated_image_set[idx_to_poison])
#         print(perturbated_image)
        perturbated_image_set[idx_to_poison] = perturbated_image

        output = model(perturbated_image_set, fast_weights, bn_training= True)
#         print(torch.nn.functional.softmax(output, dim=1).data[query_attack_idx])
#         print(torch.nn.functional.softmax(output, dim=1).data)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[query_attack_idx][target]
        max_probability = torch.max(torch.nn.functional.softmax(output, dim=1).data[query_attack_idx])
#         print(torch.nn.functional.softmax(output, dim=1).data[query_attack_idx][0])
#         print(target_probability)
#         print(max_probability)
    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch, count

# Initialize the patch
def patch_initialization(patch_type='rectangle', image_size=(3, 84, 84), noise_percentage=0.03):
    if patch_type == 'rectangle':
        # Takes dimensions of image, using percentage of "noise" gives the total
        # area of the patch. Taking sqrt gives size of one side
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)

        # Makes a random square matrix with same amount of layers as image
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

# Generate the mask and apply the patch
def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 84, 84)):
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
#     for i, set_ in enumerate(test_loader):
    for i, set_ in enumerate(test_loader):
        x_spt, y_spt, x_qry, y_qry = set_
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0), y_spt.squeeze(0), \
                                 x_qry.squeeze(0), y_qry.squeeze(0)
        
        if(i != 0):
            continue
#         for j in range(len(x_qry)):
            
#             image = x_qry[j]
#             label = y_qry[j]
            
        test_total += len(y_qry)#label.shape[0]
#             assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        # image = image.cuda()
        # label = label.cuda()
        output = model(x_qry)
        _, predicted = torch.max(output.data, 1)
        
        for i in range(len(x_qry)):
            if predicted[i] != y_qry[i] and predicted[i].data.cpu().numpy() != 0:
                test_actual_total += 1
                applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 84, 84))
                applied_patch = torch.from_numpy(applied_patch)
                mask = torch.from_numpy(mask)
                perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
#                 perturbated_image = perturbated_image.cuda()
                output = model(perturbated_image)
                _, predicted = torch.max(output.data, 1)
                if predicted[i].data.cpu().numpy() == target:
                    test_success += 1
#     break
    return test_success / test_actual_total
