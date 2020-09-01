import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import PIL
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from os import listdir
from os.path import join, splitext, basename
import glob

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
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        ImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def only_resize():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-90, 90), mode='symmetric'),
            iaa.GammaContrast((0.5, 1.5)),
            iaa.Multiply((0.5, 1.5)),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])
      
    def __call__(self, img):
        img = np.array(img)
        return PIL.Image.fromarray(self.aug.augment_image(img))

class Food11Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, input_transform, is_train=False):
        super(Food11Dataset, self).__init__()
        path_pattern = image_dir + '/**/*.*'
        files_list = glob.glob(path_pattern, recursive=True)
        self.datapath = image_dir
        self.image_filenames = []
        self.num_per_classes = [0] * 11

        for file in files_list:
            if is_image_file(file):
                self.image_filenames.append(file)
                class_name = int(basename(file).split("_")[0])
                self.num_per_classes[class_name] += 1
        self.input_transform = input_transform

    def __getitem__(self, index):
        # TODO [Lab 2-1] Try to embed third-party augmentation functions into pytorch flow
        input_file = self.image_filenames[index]
        input = load_img(input_file)
        if self.input_transform:
            input = self.input_transform(input)
        label = basename(self.image_filenames[index])
        label = int(label.split("_")[0])
        return input, label

    def __len__(self):
        return len(self.image_filenames)

    def show_details(self):
        s = sum(self.num_per_classes)
        for key in range(len(self.num_per_classes)):
            print("{:<8}#{:<20}#{:<12}#{:<6}".format(
                key,
                code2names[key],
                self.num_per_classes[key],
                self.num_per_classes[key] / s,
            ))

    def augmentation(self, wts):
        assert len(wts) == 11

        accumulated = [0] * 11
        new_fn = []
        for fn in self.image_filenames:
            class_index = int(basename(fn).split("_")[0])
            accumulated[class_index] += (wts[class_index] - 100)
            while True:
                if(accumulated[class_index] >= 100):
                    new_fn.append(fn)
                    self.num_per_classes[class_index] += 1
                    accumulated[class_index] -= 100
                    continue
                elif(accumulated[class_index] <= -100):
                    accumulated[class_index] += 100
                    self.num_per_classes[class_index] -= 1
                    break
                new_fn.append(fn)
                break

        self.image_filenames = new_fn
        