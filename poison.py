# -*- coding: utf-8 -*-
"""A file to store all poisoning APIs"""

import torch
import numpy as np
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