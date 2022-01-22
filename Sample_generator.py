import numpy as np
import matplotlib
import sys
import pylab
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from keras.datasets import mnist
from cgan import CGAN
import os
import copy
import time
import random

import warnings

warnings.filterwarnings("ignore")

def unique_index(L,f,sample_idx):
    return [i for (i,value) in enumerate(L) if value==f and i in sample_idx]

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out
def value_assign(x,j,y,digit):
    for index, element in np.ndenumerate(x[j][0]):
        x[j][0][index]=copy.deepcopy(y[index])
    for index, element in np.ndenumerate(x[j][1]):
        x[j][0][index]=copy.deepcopy(digit)
    return x

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label

# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(        
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(        
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(        
            nn.Linear(256, 1), 
            nn.Sigmoid())

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer2(x2)
        x = self.layer4(x3) 
        return x

# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(256, 784),
            nn.Tanh())

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = self.layer3(x2)        
        return x
    
def data_assign(args,datatrain,dict_users,gener_data, dict_users_gener):
    if args.ratio_realdata==1:
        dataset_train_fin = copy.deepcopy(datatrain)
        dict_users_fin = copy.deepcopy(dict_users)
    elif args.ratio_realdata==0:
        dataset_train_fin = copy.deepcopy(gener_data)
        dict_users_fin = copy.deepcopy(dict_users_gener)
    else:
        dataset_train_fin = []
        j = 0
        dict_users_fin = {}
        for idx in range(args.num_users):
            idx_begin = j
            num_real = int(args.ratio_realdata*len(dict_users[idx]))
            num_gener = args.num_items_train-num_real
            for s in range(num_real):
                dataset_train_fin.append(datatrain[list(dict_users[idx])[s]])
                j += 1
            gener_list = list(dict_users_gener[idx])
            random.shuffle(gener_list)
            for s in range(num_gener):
                dataset_train_fin.append(gener_data[gener_list[s]])
                j += 1
            dict_users_fin[idx] = set(range(idx_begin,j))
    return dataset_train_fin, dict_users_fin

def generate_sample(args,datatrain,dict_users):
    
    z_dimension = 100
    digits = [0,1,2,3,4,5,6,7,8,9]
    use_gpu = torch.cuda.is_available()

    if not os.path.exists('./image_gener'):
        os.mkdir('./image_gener')
    gener_data = []
    j, dict_users_gener = 0, {}
    for idx in range(args.num_users):
        print('Generate user data:{}'.format(idx))
        idx_begin = j
        
        if args.c_gan == True:
            
            (X_train_org, y_train_org), (_, _) = mnist.load_data()
            
            X_train, y_train = [], []
            for i in list(dict_users[idx]):
                X_train.append(X_train_org[i])
                y_train.append(y_train_org[i])
            y_train=np.array(y_train)    
            X_train=np.array(X_train)
            cgan = CGAN()
            fake_img = cgan.train(
                X_train = X_train,
                y_train = y_train, 
                epochs = args.num_epochs_gener, 
                batch_size = 32, 
                sample_interval = args.num_epochs_gener, 
                user_idx = idx)
            
            fake_images = torch.Tensor(fake_img.reshape(len(fake_img),1,28,28))
            # y_train = torch.from_numpy(y_train)
            # fake_img = torch.Tensor(fake_img)
            # fake_images = to_img(fake_img)
            for i in range(len(fake_images)):
                gener_data.append(tuple([fake_images[i],int(y_train[i])]))
                j += 1
        else:
            sample_idx = dict_users[idx]
            # Data loader
            num_labels = datatrain.train_labels.numpy()
            classes = np.unique(num_labels)
            idx_class = []
            for m in range(len(classes)):
                idx_class.append(unique_index(num_labels, classes[m],sample_idx))
    
            for digit in digits:
                D = discriminator()
                G = generator()
                if torch.cuda.is_available():
                    D = D.cuda()
                    G = G.cuda()
                # Binary cross entropy loss and optimizer
                criterion = nn.BCELoss()
                d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
                g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
                
                dataloader = torch.utils.data.DataLoader(
                    dataset = DatasetSplit(datatrain, idx_class[digit]), batch_size=len(idx_class[digit]), shuffle=True)
                
                # Start training
                for epoch in range(args.num_epochs_gener):
                    
                    for i, (img, label) in enumerate(dataloader):
                        num_img = img.size(0)
                        # train discriminator
                        img = img.view(num_img, -1)
                        real_img = Variable(img).cuda()
                        real_label = Variable(torch.ones(num_img)).cuda()
                        fake_label = Variable(torch.zeros(num_img)).cuda()
                
                        # compute loss of real_img
                        real_out = D(real_img)
                                
                        d_loss_real = criterion(real_out, real_label)
                        
                        #d_loss_real=d_loss_real+noise
                        real_scores = real_out  # closer to 1 means better
                
                        # compute loss of fake_img
                        z = Variable(torch.randn(num_img, z_dimension)).cuda()
                        fake_img = G(z)
                        fake_out = D(fake_img)
                        d_loss_fake = criterion(fake_out, fake_label)
                        fake_scores = fake_out  # closer to 0 means better
                
                        # bp and optimize
                        d_loss = d_loss_real + d_loss_fake
                        d_optimizer.zero_grad()
                        d_loss.backward()
                    
                        d_optimizer.step()
                
                        # train generator
                        # compute loss of fake_img
                        z = Variable(torch.randn(num_img, z_dimension)).cuda()
                        fake_img = G(z)
                        output = D(fake_img)
                        g_loss = criterion(output, real_label)
                        # bp and optimize
                        g_optimizer.zero_grad()
                        g_loss.backward()
                        g_optimizer.step()
                
                        if (i + 1) % 5 == 0:
                            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                                  'D real: {:.6f}, D fake: {:.6f}'.format(
                                      epoch, args.num_epochs_gener, d_loss.item(), g_loss.item(),
                                      real_scores.data.mean(), fake_scores.data.mean()))
                    if epoch == 0:
                        real_images = to_img(real_img.cpu().data)
                        save_image(real_images, './image_gener/record-{}.png'.format(digit, epoch + 1))
                
                    if (epoch+1)%100 == 0:  
                        fake_images = to_img(fake_img.cpu().data)
                        save_image(fake_images, './image_gener/fake_images_digit-{}-{}.png'.format(digit, epoch + 1))
            for i in range(len(fake_images)):
                gener_data.append(tuple([fake_images[i],digit]))
                j += 1
        dict_users_gener[idx] = set(range(idx_begin,j))
    return gener_data, dict_users_gener