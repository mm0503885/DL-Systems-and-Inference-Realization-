#!/usr/bin/env python
# coding: utf-8

# In[16]:


# !/usr/bin/env python
# coding: utf-8

# In[78]:


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



#To determine if your system supports CUDA
print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)

#Also can print your current GPU id, and the number of GPUs you can use.
print("Our selected device: ", torch.cuda.current_device())
print(torch.cuda.device_count(), " GPUs is available")



print('==> Preparing dataset..')

"""1.1"""
# The output of torchvision datasets are PILImage images of range [0, 1]
# We transform them to Tensor type
# And normalize the data
# Be sure you do same normalization for your train and test data

#The transform function for train data


data_transforms = {
    'skewed_training': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),    
    'evaluation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


"""1.2""" 

data_dir = 'food11re'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['skewed_training','validation','evaluation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=2)
              for x in ['skewed_training','validation','evaluation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['skewed_training','validation','evaluation']}
class_names = image_datasets['skewed_training'].classes


print('==> Building model..')

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_values = []
    acc_values = []
    patient = 5
    j = 0
    end = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['skewed_training', 'validation']:
            if phase == 'skewed_training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'skewed_training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'skewed_training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'skewed_training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'validation':
                loss_values.append(epoch_loss)
                acc_values.append(epoch_acc)
                # deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    j = 0
                else:
                    j += 1
                    if j == patient:
                        end = 1

        print()
        if end == 1:
            break
    plt.figure()
    plt.subplot(1,2,1)
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("lr=0.001, step_size=7(loss)")
    plt.plot(np.array(loss_values), 'r')
    plt.subplot(1,2,2)
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("lr=0.001, step_size=7(accuracy)")
    plt.plot(np.array(acc_values), 'r')
    plt.show()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 11)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


PATH = 'lab1\lab1_model.pht'
torch.save(model_ft.state_dict(), PATH)


class_correct = list(0. for i in range(11))
class_total = list(0. for i in range(11))
correct_top3 = 0.
with torch.no_grad():
    for data in dataloaders['evaluation']:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model_ft(images)
        _, predicted = torch.max(outputs, 1)
        _, predicted_top3 = outputs.topk(3, dim=1, largest=True, sorted=True)
        c = (predicted == labels).squeeze()
        c_top3_0 = (predicted_top3[:,0] == labels).squeeze()
        c_top3_1 = (predicted_top3[:,1] == labels).squeeze()
        c_top3_2 = (predicted_top3[:,2] == labels).squeeze()
        if(labels.size()==torch.Size([4])):
            for i in range(4):
                class_correct[labels[i]] += c[i].item()
                class_total[labels[i]] += 1
                correct_top3 += c_top3_0[i].item()
                correct_top3 += c_top3_1[i].item()
                correct_top3 += c_top3_2[i].item()
        else:
            for i in range(3):
                class_correct[labels[i]] += c[i].item()
                class_total[labels[i]] += 1
                correct_top3 += c_top3_0[i].item()
                correct_top3 += c_top3_1[i].item()
                correct_top3 += c_top3_2[i].item()

total_correct = 0.
for i in range(11):
    total_correct+=class_correct[i]

print('Test set: Top1 Accuracy: %d/%d (%2d %%) , Top3 Accuracy: %d/%d (%2d %%)' % (
    total_correct,dataset_sizes['evaluation'], 100 * total_correct / dataset_sizes['evaluation'],
    correct_top3,dataset_sizes['evaluation'], 100 * correct_top3 / dataset_sizes['evaluation']))
    
for i in range(11):
    print('class %s : %d/%d %2d %%' % (
        class_names[i],class_correct[i],class_total[i], 100 * class_correct[i] / class_total[i]))


# In[ ]:




