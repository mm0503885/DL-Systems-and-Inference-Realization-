import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import PIL
from PIL import Image
from os import listdir
from os.path import join, splitext, basename
import glob
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
Image.MAX_IMAGE_PIXELS = 1000000000

test_transforms=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
def show_layer_sparsity(net):
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]

    parameters_to_prune = []
    parameters_to_prune.append((net.conv1, 'weight', 'conv1'))
    for i, l in enumerate(layers):
        for b in range(blocks[i]):
            parameters_to_prune.append((l[b].conv1, 'weight', 'layer'+str(i+1)+'.conv1'))
            parameters_to_prune.append((l[b].conv2, 'weight', 'layer'+str(i+1)+'.conv2'))
    parameters_to_prune.append((net.fc, 'weight', 'fc'))
    parameters_to_prune.append((net.fc, 'bias', 'fc'))

    for p in parameters_to_prune:
        if(p[1] == 'weight'):
            print((p[2]+'.weight:').ljust(22) + ('%.4f%%' % (100. * float(torch.sum(p[0].weight == 0)) / float(p[0].weight.nelement()))))
        else:
            print((p[2]+'.bias:').ljust(22) + ('%.4f%%' % (100. * float(torch.sum(p[0].bias == 0)) / float(p[0].bias.nelement()))))

def test(net, test_loader):
    correct = 0
    net.eval()
    start_time = time.time()
    for i, (inputs, labels) in enumerate(test_loader, 0):
        inputs = inputs.to(device) 
        labels = labels.to(device) 
        
        outputs = net(inputs)
        _, pred = outputs.max(1)
    
        for i in range(len(labels)):
            if(labels[i] ==  pred[i]):
                correct += 1
    latency = time.time() - start_time
    print('test accuracy: %.4f' % (100.*correct / len(test_set)))
#    print('Inference average time: %.3f ms' % (latency* 1000/len(test_set)))
#    print('FPS with overhead: %.3f' % (len(test_set)/(latency)))


test_datapath = "food11re/evaluation/"


#test_set = datasets.ImageFolder(test_datapath, test_transforms)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)

test_datapath = "food11re/evaluation/"
test_set = dataset.Food11Dataset(test_datapath, input_transform=dataset.only_resize())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

net = models.resnet34(pretrained=True)
num_ftrs = net.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
net.fc = nn.Linear(num_ftrs, 11)
net = net.to(device)
net.load_state_dict(torch.load('lab4_model.pht'))

test(net, test_loader)

amounts = [0.5, 0.66]

for a in amounts:
    blocks = [3, 4, 6, 3]
    net = models.resnet34(pretrained=False)
    num_ftrs = net.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    net.fc = nn.Linear(num_ftrs, 11)
    net = net.to(device)
    net.load_state_dict(torch.load('lab4_model.pht'))
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]

    parameters_to_prune = []
    parameters_to_prune.append((net.conv1, 'weight'))
    for i, l in enumerate(layers):
        for b in range(blocks[i]):
            parameters_to_prune.append((l[b].conv1, 'weight'))
            parameters_to_prune.append((l[b].conv2, 'weight'))
    parameters_to_prune.append((net.fc, 'weight'))
    parameters_to_prune.append((net.fc, 'bias'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=a,
    )

    print('Gobal sparsity: %f' % a)
    show_layer_sparsity(net)
    test(net, test_loader)
