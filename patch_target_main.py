import torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
from MiniImagenet import MiniImagenet
from torch.utils.data import DataLoader
from Meta_Modified import Meta
from Learner import Learner
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import patch_utils
import patch_attack_target
import random


def load_model(path, n_way = 5, cpu = True, meta_object = False):
    """
    Method Loads Model into a meta object
    """
    n_way = n_way
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
            ('linear', [n_way, 32 * 5 * 5])
        ]
    model = Learner(miniImgNet_config,3,84)
    maml = Meta(miniImgNet_config)
    
    # Below is to account for different settings of cpu and meta vs learner object
    if(cpu):
        if(not meta_object):
            maml.net.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        else:   
            maml.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    else:
        if(not meta_object):
            maml.net.load_state_dict(torch.load(path))
        else:
            maml.load_state_dict(torch.load(path))
            
    return maml

def getAccuracy(mini_test, maml):
    """
    Calling this method will return the total accurary
    """
    total_acc = 0
    for i, set_ in enumerate(mini_test):

        #unsqueeze
        support_x, support_y, query_x, query_y = set_
        support_x, support_y, query_x, query_y = support_x.squeeze(0), support_y.squeeze(0), \
                                     query_x.squeeze(0), query_y.squeeze(0)

        acc, preq_NP, fast_weights = maml.finetuning(support_x, support_y, query_x, query_y) 

        total_acc += acc[-1]

    print("Total Acc: ")
    print(total_acc/len(mini_test))
    
    return total_acc/len(mini_test)


def getConstantClassAndPatch(mini_test):
    for i, set_ in enumerate(mini_test):

        #unsqueeze
        support_x, support_y, query_x, query_y = set_
        support_x, support_y, query_x, query_y = support_x.squeeze(0), support_y.squeeze(0), \
                                     query_x.squeeze(0), query_y.squeeze(0)

        # Get a class to poison for all tests
        idx_support = (support_y == 0).nonzero(as_tuple=False)
        idx_query = (query_y == 0).nonzero(as_tuple=False)
        images_to_patch_support = support_x[idx_support]
        images_to_patch_query = query_x[idx_query]

        # Create patch and mask and convert to torch
        patch_target = patch_utils.patch_initialization()
        applied_patch_target, mask_target, x_location_target, y_location_target = patch_utils.mask_generation(patch = patch_target)
        
        
        return images_to_patch_support, images_to_patch_query, applied_patch_target, mask_target, patch_target

    
def craftAttackTargeted(mini_test, maml, images_to_patch_support, images_to_patch_query, applied_patch_target, mask_target, patch_target):
    
    """
    This class crafts the patch attack for targeted target given that each task has one class with the same support set that is the same across all tasks 
    """
    
    attempts = 0
    success = 0
    #iterate over the different tasks
    for i, set_ in enumerate(mini_test):
        
        patch = patch_target
        applied_patch = applied_patch_target
        mask = mask_target
        
        # pass over task the first task #### This could be modified so it doesn't have to do this ####
        if(i == 0):
            continue
        support_x, support_y, query_x, query_y = set_
        support_x, support_y, query_x, query_y = support_x.squeeze(0), support_y.squeeze(0), \
                                 query_x.squeeze(0), query_y.squeeze(0)

        # get random position for class
        target_class_untargeted = np.random.randint(0, 5)

        # set all the support and query
        idx_support = (support_y == target_class_untargeted).nonzero(as_tuple=False)
        idx_query = (query_y == target_class_untargeted).nonzero(as_tuple=False)

        # if task no is less than 66 only train patch on the first 10 query set 
        if(i < 66):
            query_attack_idx = idx_query[np.random.randint(0, len(idx_query)-5)]
        
        # if task no is greater than 66 test on unseen query images
        else:
            query_attack_idx = idx_query[np.random.randint(len(idx_query)-5, len(idx_query))]

        # set class support images of a given class into a random class in new support set
        for j, idx in enumerate(idx_support):
            support_x[idx] = images_to_patch_support[j]

        # get a random class that is not the class that you have just replaced
        target_class = np.random.randint(0,5)
        while(target_class == target_class_untargeted):
            target_class = np.random.randint(0,5)

        # get idx for image of the target class to poison
        idx_support_target = (support_y == target_class).nonzero(as_tuple=False)
        random_support_image_chosen_idx_of_idx = np.random.randint(0,len(idx_support_target))
        support_image_chosen_idx = idx_support_target[random_support_image_chosen_idx_of_idx]

        # applied patch and mask to torch object
        applied_patch_1 = torch.from_numpy(applied_patch)
        mask_1 = torch.from_numpy(mask)

        # poison selected image in the current task
        support_x[support_image_chosen_idx] = torch.mul(mask_1.type(torch.FloatTensor), applied_patch_1.type(torch.FloatTensor)) + torch.mul((1 - mask_1.type(torch.FloatTensor)), support_x[support_image_chosen_idx].type(torch.FloatTensor)) 
        
        
        # set query images as the constant class
        for j, idx in enumerate(idx_query):
            query_x[idx] = images_to_patch_query[j]
            
        # get acc, preq_values, and fast_weights fo the new support_x
        acc, preq_P, fast_weights = maml.finetuning(support_x, support_y, query_x, query_y) 
        

        print("Task no: ", i)
        if(i < 66):
            
            end_image, end_patch, count = patch_attack_target.patch_attack(query_x, applied_patch_target, mask_target, target_class, query_attack_idx, 0.8, maml.net, fast_weights)
            applied_patch_target = end_patch
            query_x[query_attack_idx] = torch.from_numpy(end_image)

        
            applied_patch_1 = torch.from_numpy(applied_patch_target)
            mask_1 = torch.from_numpy(mask_target)
        else:
            applied_patch_1 = torch.from_numpy(applied_patch_target)
            mask_1 = torch.from_numpy(mask_target)
            query_x[query_attack_idx] = torch.mul(mask_1.type(torch.FloatTensor), applied_patch_1.type(torch.FloatTensor)) + torch.mul((1 - mask_1.type(torch.FloatTensor)), query_x[query_attack_idx].type(torch.FloatTensor)) 

        
        preq = maml.net(query_x, fast_weights, bn_training=False)
        _, predicted = torch.max(preq.data, 1)
#         __, predicted_true = torch.max(preq_P.data, 1)
        
    #     print(i)
    #     print("T: ", target_class, "; UT: ", target_class_untargeted)
    #     print("idx: ", query_attack_idx)
    #     print("predcited: ", predicted[query_attack_idx])
    #     print("actual: ", query_y[query_attack_idx])
    #     print(predicted[query_attack_idx]==query_y[query_attack_idx])
    #     print(count)

        if(i >= 66):
            attempts += 1

        if(target_class == predicted[query_attack_idx] and i >= 66):
            success+=1
            print("Success at: ", i)
            print("Current Success Rate: ", success/attempts)

    print(success/attempts)

    return success/attempts
    
    
if __name__ == '__main__':
    
    model_path = "MiniImagenet_n_way_5_k_shot_5_acc_0.60546875_dateMMDDYYY_10272021"

    maml = load_model(model_path)
    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=5, k_shot=5,
                             k_query=15,
                             batchsz=100, resize=84)
    
    images_to_patch_support, images_to_patch_query, applied_patch_target, mask_target, patch_target = getConstantClassAndPatch(mini_test)
    success_rate = craftAttackTargeted(mini_test, maml, images_to_patch_support, images_to_patch_query, applied_patch_target, mask_target, patch_target)
    print("success rate: ", success_rate)