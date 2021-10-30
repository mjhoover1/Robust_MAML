# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 18:50:43 2021

@author: bob12
"""

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

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