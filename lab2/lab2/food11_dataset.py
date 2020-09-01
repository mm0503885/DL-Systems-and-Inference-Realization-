from os import listdir
from os.path import join, splitext, basename
import glob

import torch.utils.data as data
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from PIL import Image

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
    
    ''' TODO [Lab 2-1]
    #please add a new function "augmentation(self, wts)"
    #it can change the number of data according to weight of each category
    #"weight" represents the ratio comparing with the amount of original set
    #if the weight > 100, we create new data by copying
    #if the weight < 100, we will delete the original data
    #[hint]you only need to edit the "self.image_filenames" 

    wts = [ 125, 80, 25, 100, 200, 800, 80, 60, 40, 150, 1000 ]
    def augmentation(self)
        if is_train:
            pass

    def augmentation(self, wts)
        if is_train:
            pass

    ODOT [Lab 2-1]'''
     

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
    train_datapath = "/food11re/skewed_training"
    valid_datapath = "/food11re/validation"
    test_datapath = "/food11re/evaluation"

    train_dataset = Food11Dataset(train_datapath, is_train=True)
    valid_dataset = Food11Dataset(valid_datapath, is_train=False)
    test_dataset = Food11Dataset(test_datapath, is_train=False)

    ''' For [Lab 2-1] debugging
    train_dataset.augmentation()
    wts = [ 125, 80, 25, 100, 200, 800, 80, 60, 40, 150, 1000 ]
    train_dataset.augmentation(wts)
    '''

    print("----------------------------------------------------------------------------------")
    print("Dataset bf. loading - ", train_datapath)
    print(train_dataset.show_details())

    print("----------------------------------------------------------------------------------")
    print("Dataset bf. loading - ", valid_datapath)
    print(valid_dataset.show_details())

    print("----------------------------------------------------------------------------------")
    print("Dataset bf. loading - ", test_datapath)
    print(test_dataset.show_details())

    train_loader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=8, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=4, batch_size=8, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=4, batch_size=8, shuffle=False)

    data_loading(train_loader, train_dataset)
    data_loading(valid_loader, valid_dataset)
    data_loading(test_loader, test_dataset)

if __name__ == '__main__':
    main()
