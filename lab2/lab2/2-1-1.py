

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
import sys

from imgaug import augmenters as iaa
import imgaug as ia



class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Resize((224, 224)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

tfs = transforms.Compose([
    ImgAugTransform(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



data_dir = 'food11re'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          tfs)
                  for x in ['skewed_training','validation','evaluation']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=24,
                                             shuffle=True, num_workers=0)
              for x in ['skewed_training','validation','evaluation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['skewed_training','validation','evaluation']}
class_names = image_datasets['skewed_training'].classes

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
dataiter = iter(dataloaders['skewed_training'])
images, labels = dataiter.__next__()

# show images
imshow(torchvision.utils.make_grid(images))





