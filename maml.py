# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:40:43 2021

@author: bob12
"""

import torch
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear)
from collections import OrderedDict
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters



def conv3x3(in_channels, out_channels):
  return MetaSequential(
      MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1),
      MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
      nn.ReLU(),
      nn.MaxPool2d(2)
  )

class ConvolutionalNeuralNetwork(MetaModule):
  def __init__(self, in_channels, out_features, hidden_size=64):
    super(ConvolutionalNeuralNetwork, self).__init__()
    self.in_channels = in_channels
    self.out_features = out_features
    self.hidden_size = hidden_size

    self.features = MetaSequential(
        conv3x3(in_channels, hidden_size),
        conv3x3(hidden_size, hidden_size),
        conv3x3(hidden_size, hidden_size),
        conv3x3(hidden_size, hidden_size)
    )

    self.classifier = MetaLinear(hidden_size, out_features)

  def forward(self, inputs, params=None):
    features = self.features(inputs, params=self.get_subdict(params, 'features'))
    features = features.view((features.size(0), -1))
    logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
    return logits

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.
        
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train(folder, device, num_shots=5, num_ways=5, batch_size=16, num_batches=100, num_workers=1, hidden_size=64, step_size=.4, first_order=True, output_folder=None):
  dataset = omniglot(folder, shots=num_shots, ways=num_ways, shuffle=True, test_shots=15, 
                     meta_train=True, download=True)
  dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                   num_workers=num_workers)
  model = ConvolutionalNeuralNetwork(1, num_ways, hidden_size=hidden_size)
  model.to(device=device)
  model.train()
  meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  # Training loop
  with tqdm(dataloader, total=num_batches) as pbar:
    for batch_idx, batch in enumerate(pbar):

      model.zero_grad()

      train_inputs, train_targets = batch['train']
      train_inputs = train_inputs.to(device=device)
      train_targets = train_targets.to(device=device)

      test_inputs, test_targets = batch['test']
      test_inputs = test_inputs.to(device=device)
      test_targets = test_targets.to(device=device)
      
      outer_loss = torch.tensor(0., device=device)
      accuracy = torch.tensor(0., device=device)
      for task_idx, (train_input, train_target, test_input,
                     test_target) in enumerate(zip(train_inputs, train_targets,
                                                   test_inputs, test_targets)):
                       train_logit = model(train_input)
                       inner_loss = F.cross_entropy(train_logit, train_target)

                       model.zero_grad()
                       params = gradient_update_parameters(model,
                                                           inner_loss,
                                                           step_size=step_size,
                                                           first_order=first_order)
                       
                       test_logit = model(test_input, params=params)
                       outer_loss += F.cross_entropy(test_logit, test_target)

                       with torch.no_grad():
                         accuracy += get_accuracy(test_logit, test_target)
      outer_loss.div_(batch_size)
      accuracy.div_(batch_size)

      outer_loss.backward()
      meta_optimizer.step()

      pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
      if batch_idx >= num_batches: break

  # Save model
  if output_folder is not None:
    filename = os.path.join(output_folder, 'maml_omniglot_{0}shot_{1}way.th'.format(
        num_shots, num_ways))
    with open(filename, 'wb') as f:
      state_dict = model.state_dict()
      torch.save(state_dict, f)

if __name__ == '__main__':
    device = get_default_device()
    device
    
    
    
    logger = logging.getLogger('example 1')
    train('data', device, output_folder='.')