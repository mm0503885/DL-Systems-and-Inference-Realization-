import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as trans
import torchvision.models as models
import matplotlib.pyplot as plt
#import dxchange
#from xlearn.transform import train
#from xlearn.transform import model

import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from progressbar import *

from imgaug import augmenters as iaa
import PIL
from PIL import Image

import time
import datetime


batch_size = 32
criterion = torch.nn.CrossEntropyLoss()
train_on_gpu = torch.cuda.is_available()
n_epochs = 100
lr = 0.001

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
	        iaa.Sometimes(0.2, iaa.GammaContrast((0.5, 2.0)))
            # iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

train_datapath = "/home/mel/Andrew/food11re/skewed_training"
valid_datapath = "/home/mel/Andrew/food11re/validation"
test_datapath = "/home/mel/Andrew/food11re/evaluation"

transform_train = trans.Compose([
    ImgAugTransform(),
    lambda x: PIL.Image.fromarray(x),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.Resize(size=(224,224)),
    trans.ToTensor(),
    trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = trans.Compose([
    trans.Resize(size=(224,224)),
    trans.ToTensor(),
    trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                   
])

data_train = datasets.ImageFolder(train_datapath, transform=transform_train)
data_valid = datasets.ImageFolder(valid_datapath, transform=transform_test)
data_test  = datasets.ImageFolder(test_datapath,  transform=transform_test)

weight_list = [0.67, 5.35, 0.42, 1.12, 1.28, 0.50, 5.03, 7.99, 1.30, 0.44, 18.99]
weights = []
for _, label in data_train:
    weights.append(weight_list[label])
w_sampler = torch.utils.data.WeightedRandomSampler(weights, len(data_train), replacement=True)

train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, num_workers=4, sampler=w_sampler)
#train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader  = torch.utils.data.DataLoader(data_test,  batch_size=batch_size, shuffle=True, num_workers=4)

classes = list(i for i in range(11))
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(2)

savepath_ES   = "./model/model3_1c_5018pre.pth.tar"
savepath_last = "./model/model3_1c_5018pre_last.pth.tar"

# load pretrained model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 11)

model_t = models.resnet50(pretrained=True)
num_ftrs_t = model_t.fc.in_features
model_t.fc = nn.Linear(num_ftrs_t, 11)
checkpoint = torch.load("./model/model_res50.pth.tar")
model_t.load_state_dict(checkpoint['model_state_dict'])


if os.path.isfile(savepath_ES):
    checkpoint = torch.load(savepath_ES)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model_t.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model_t = model_t.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum= 0.9)
    #optimizer_t = torch.optim.SGD(model_t.parameters(), lr = lr, momentum= 0.9)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_to_start = checkpoint['epoch'] + 1
    valid_loss_min = checkpoint['loss']
else:
    epoch_to_start = 1
    model = model.to(device)
    model_t = model_t.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum= 0.9)
    #optimizer_t = torch.optim.SGD(model_t.parameters(), lr = lr, momentum= 0.9)
    valid_loss_min = np.Inf # track change in validation loss

print('epoch to start:', epoch_to_start)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

# Early stopping coefficient
patience = 25
j = 0
# plotting curves
train_loss_plot = []
train_acc_plot = []
valid_loss_plot = []
valid_acc_plot = []
def run(arg):
    StartTime = time.time()
    for epoch in range(epoch_to_start, n_epochs+1):
    
        correct_train, correct_valid = 0, 0
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        model_t.train()
        # progress = ProgressBar()
        # for data, target in progress(train_loader):
        for data, target in train_loader:
            if train_on_gpu:
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            output_t = model_t(data)
            kl = kl_divergence(output_t, output)
            _, pred = output.max(1)
            correct_train += pred.eq(target).sum().item()
    
            loss_s = criterion(output, target)
            loss = (1-0.5)*loss_s + 0.5*kl
            loss.backward()
            optimizer.step()
            #optimizer_t.step()
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        model_t.eval()
        for data, target in valid_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            output_t = model_t(data)
            kl = kl_divergence(output_t, output)
            _, pred = output.max(1)
            correct_valid += pred.eq(target).sum().item()
            loss_s = criterion(output, target)
            loss = (1-0.5)*loss_s + 0.5*kl
            valid_loss += loss.item()*data.size(0)
    
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
            
        # print training/validation statistics
        valid_acc = 100.*correct_valid/len(data_valid)
        print('Epoch: {} \tTraining Loss: {:.6f}({:.2f}%) \tValidation Loss: {:.6f}({:.4f}%)'.format(
            epoch, train_loss, 100.*correct_train/len(data_train), valid_loss, valid_acc))
        train_loss_plot.append(train_loss)
        train_acc_plot.append(100.*correct_train/len(data_train))
        valid_loss_plot.append(valid_loss)
        valid_acc_plot.append(valid_acc)
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min and j < patience:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, valid_loss))
            valid_loss_min = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'acc': valid_acc
                }, savepath_ES)
            j = 0
        else:
            j += 1
    
    EndTime = time.time()
    print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime-StartTime)))))
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'acc': valid_acc
                }, savepath_last)
    
    # plot training, validation curves
    plt.figure()
    plt.plot(range(1, n_epochs+1), train_loss_plot, 'b', label='train')
    plt.plot(range(1, n_epochs+1), valid_loss_plot, 'g', label='valid')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig("./lab3-1_c_5018pre_loss.png")
    plt.close()
    
    plt.figure()
    plt.plot(range(1, n_epochs+1), train_acc_plot, 'b', label='train')
    plt.plot(range(1, n_epochs+1), valid_acc_plot, 'g', label='valid')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.legend(loc='upper right')
    plt.savefig("./lab3-1_c_5018pre__acc.png")
    plt.close()
    
    # track test loss
    models_to_test = [savepath_ES, savepath_last]
    
    for savepath in models_to_test:
        checkpoint = torch.load(savepath)
        model.load_state_dict(checkpoint['model_state_dict'])
    
        test_loss = 0.0
        class_correct = list(0. for i in range(11))
        class_total = list(0. for i in range(11))
    
        pred_tot = []
        model.eval()
        i=1
        # iterate over test data
        # print(len(test_loader))
        for data, target in test_loader:
            i=i+1
            # if len(target)!=batch_size:
            #     continue
                
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)    
            pred_tot.extend(pred.tolist()) 
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(len(target)):       
              label = target.data[i]
              class_correct[label] += correct[i].item()
              class_total[label] += 1
                
        # average test loss
        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))
    
        for i in range(11):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
return 100. * np.sum(class_correct) / np.sum(class_total)
def params_loader():
    '''Get parameters.'''
    parser = ArgumentParser(description='ResNet18(pretrained) on food11 skewed')
    parser.add_argument('--batch-size', type=int, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--input-size', type=int,
                        help='input size (Resize to)')
    parser.add_argument('--momentum', type=float, metavar='M',
                        help='SGD momentum')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')

    args, _ = parser.parse_known_args()
    params = {k: v for k, v in vars(args).items() if v is not None}
    return params


if __name__ == '__main__':
    args = param_loader()
    acc1 = run(args)