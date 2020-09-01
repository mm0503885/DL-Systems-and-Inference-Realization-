#!/usr/bin/env python
# coding: utf-8

# In[15]:


from os import listdir
from os.path import join, splitext, basename
import glob

import torch.utils.data as data
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.data import WeightedRandomSampler
import numpy as np
import random

code2names = {
    0:"Bread",
    1:"Dairy_product",
    2:"Dessert",
    3:"Egg",
    4:"Fried_food",
    5:"Meat",
    6:"Noodles",
    7:"Rice",
    8:"Seafood",
    9:"Soup",
    10:"Vegetable_fruit"
}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath)
    return img

def input_transform():
    return trans.Compose([
        trans.Resize((224, 224)),
        trans.ToTensor(),
    ])

class Food11Dataset(data.Dataset):
    def __init__(self, image_dir, 
                 input_transform=input_transform, is_train=False):
        super(Food11Dataset, self).__init__()
        path_pattern = image_dir + '/**/*.*'
        files_list = glob.glob(path_pattern, recursive=True)
        self.datapath = image_dir
        self.image_filenames = []
        self.num_per_classes = {}
        for file in files_list:
            if is_image_file(file):
                self.image_filenames.append(file)
                class_name = int(basename(file).split("_")[0])
                if class_name in self.num_per_classes:
                    self.num_per_classes[class_name] += 1
                else:
                    self.num_per_classes[class_name] = 1
        self.input_transform = input_transform

    def __getitem__(self, index):
        # TODO [Lab 2-1] Try to embed third-party augmentation functions into pytorch flow
        input_file = self.image_filenames[index]
        input = load_img(input_file)
        if self.input_transform:
            input = self.input_transform()(input)
        label = basename(self.image_filenames[index])
        label = int(label.split("_")[0])
        return input, label

    def __len__(self):
        return len(self.image_filenames)

    def show_details(self):
        for key in sorted(self.num_per_classes.keys()):
            print("{:<8}|{:<20}|{:<12}".format(
                key,
                code2names[key],
                self.num_per_classes[key]
            ))
    
    def augmentation(self, wts):
        class_start = np.zeros(11,dtype='int')
        p = 0
        for i in range(11):
            if i < 2:
                class_start[i] = p
                p += self.num_per_classes[i]
            elif i == 2:
                class_start[10] = p
                p += self.num_per_classes[10]
            else:
                class_start[i-1] = p
                p += self.num_per_classes[i-1]
        for i in range(11):
            num_to_aug = round(self.num_per_classes[i]*wts[i]/100)
            if wts[i] > 100:
                num_to_create = num_to_aug - self.num_per_classes[i]
                for _ in range(num_to_create):
                    random_pick = random.randrange(self.num_per_classes[i])
                    aug_filename = self.image_filenames[class_start[i]+random_pick]
                    self.image_filenames.insert(class_start[i],aug_filename)
                if i < 2:
                    for k in range(i+1,11):
                        class_start[k] += num_to_create
                elif i == 10:
                    for k in range(2,10):
                        class_start[k] += num_to_create
                else:
                    for k in range(i+1,10):
                        class_start[k] += num_to_create
            elif wts[i] < 100:
                num_to_delete = self.num_per_classes[i] - num_to_aug
                for j in range(num_to_delete):
                    random_pick = random.randrange(self.num_per_classes[i]-j)
                    del self.image_filenames[class_start[i]+random_pick]
                if i < 2:
                    for k in range(i+1,11):
                        class_start[k] -= num_to_delete
                elif i == 10:
                    for k in range(2,10):
                        class_start[k] -= num_to_delete
                else:
                    for k in range(i+1,10):
                        class_start[k] -= num_to_delete

     

def data_loading(loader, dataset):

    num_per_classes = {}
    for batch_idx, (data, label) in enumerate(loader):
        for l in label:
            if l.item() in num_per_classes:
                num_per_classes[l.item()] += 1
            else:
                num_per_classes[l.item()] = 1

    print("----------------------------------------------------------------------------------")
    print("Dataset - ", dataset.datapath)
    print("{:<20}|{:<15}|{:<15}".format("class_name", "bf. loading", "af. loading"))
    for key in sorted(num_per_classes.keys()):
        print("{:<20}|{:<15}|{:<15}".format(
            code2names[key],
            dataset.num_per_classes[key],
            num_per_classes[key]
        ))

def main():
    train_datapath = "food11re/skewed_training"
    valid_datapath = "food11re/validation"
    test_datapath = "food11re/evaluation"

    train_dataset = Food11Dataset(train_datapath, is_train=True)
    valid_dataset = Food11Dataset(valid_datapath, is_train=False)
    test_dataset = Food11Dataset(test_datapath, is_train=False)
    
    weights_per_class = [1/train_dataset.num_per_classes[i] for i in range(11)]
    weight = np.zeros(len(train_dataset))
    class_idx = [0,1,10,2,3,4,5,6,7,8,9]
    start = 0
    num_train_data = 10000
    for i in class_idx:
        weight[start:start+train_dataset.num_per_classes[i]] = weights_per_class[i]
        start += train_dataset.num_per_classes[i]
    
    train_sampler = WeightedRandomSampler(weight,num_train_data)
    train_sampler2 = RandomSampler(train_dataset,replacement=True, num_samples=num_train_data)
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=8, sampler=train_sampler)
    train_loader2 = DataLoader(dataset=train_dataset, num_workers=0, batch_size=8, sampler=train_sampler2)
    
    data_loading(train_loader, train_dataset)
    data_loading(train_loader2, train_dataset)

if __name__ == '__main__':
    main()


# In[ ]:




