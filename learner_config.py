# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 17:43:38 2021

@author: bob12
"""

n_ways = 5
# [ch_in, ch_out, kernelsz, kernelsz, stride, padding] 256, 128, 64, 64

config_small = [
    ('conv2d', [32, 3, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('conv2d', [32, 32, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('conv2d', [32, 32, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('conv2d', [32, 32, 2, 2, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('flatten', []),
    ('linear', [n_ways, 4*32])
]
config_big = [
    ('conv2d', [256, 3, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [256]),
    ('conv2d', [128, 256, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [128]),
    ('conv2d', [64, 128, 3, 3, 2, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('conv2d', [64, 64, 2, 2, 1, 0]),
    ('relu', [True]),
    ('bn', [64]),
    ('flatten', []),
    ('linear', [n_ways, 4*64])
]

miniImgNet_config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [n_ways, 32 * 5 * 5])
    ]