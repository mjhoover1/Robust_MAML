# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 17:44:52 2021

@author: bob12
"""

## Standard libraries
import random

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
#TODO: %matplotlib inline ???
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## PyTorch
import torch

#TODO: Where does this go?
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
"""

def createClassSplits(all_imgs, all_labels):#, train_classes, test_classes, val_classes):
    """
    Inputs:
        imgs - Numpy array of shape [N,32,32,3] containing all images.
        targets - PyTorch array of shape [N] containing all labels.
        k-shot -

    """

    classes = torch.unique(all_labels).tolist()
    data_by_classes = []
    # print(classes)

    for class_ in classes:

        holder_array = []
        if(class_ % 10 == 0): print(class_)
        idx_array = [i for i, x in enumerate(all_labels) if x == class_]

        labels_for_class = [all_labels[i] for i in idx_array]
        img_for_class = [all_imgs[i] for i in idx_array]
        combined_by_class =[]

        combined_by_class = [[img_for_class[i],labels_for_class[i]] for i in range(len(img_for_class))]

        data_by_classes.append(combined_by_class)

    return data_by_classes

def sampleTasks(data_by_classes, k_shots, n_way, k_qry):
    task_extra = random.sample(data_by_classes, n_way)
    task_ss = []
    task_q = []
    for by_class in task_extra:

        samples = random.sample(by_class, 2*k_shots)

        task_ss = task_ss + samples[:k_shots]
        task_q = task_q +samples[k_shots:k_shots+k_qry]

    random.shuffle(task_ss)
    random.shuffle(task_q)

    return task_ss, task_q[:k_qry*k_shots]
