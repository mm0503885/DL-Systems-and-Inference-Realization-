#!/usr/bin/env python
# coding: utf-8

# In[15]:


import torch
from torchvision import models
from thop.profile import profile


print('Lab 1-2:\n')
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = models.resnet50().to(device)
model2 = models.mobilenet_v2().to(device)
dsize = (1, 3, 224, 224)
inputs = torch.randn(dsize).to(device)
print('resnet50:')
total_MACs, total_params = profile(model, (inputs,))
print("Total params: %.2fM" % (total_params / (1000 ** 2)))
print("Total MACs: %.2fM\n" % (total_MACs / (1000 ** 2)))
print('mobilenet_v2:')
total_MACs2, total_params2 = profile(model2, (inputs,))
print("Total params: %.2fM" % (total_params2 / (1000 ** 2)))
print("Total MACs: %.2fM" % (total_MACs2 / (1000 ** 2)))


# In[7]:


print(models)


# In[ ]:




