#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
import matplotlib
import sys
import pylab
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
import math
import numpy as np
import random
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import autograd
from tensorboardX import SummaryWriter
from sympy import solve
from sympy.abc import y
import sympy as sy
from scipy import optimize

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP1, CNNMnist, CNN_test
from averaging import average_weights
from Sample_generator import generate_sample, data_assign


if __name__ == '__main__':    
    # return the available GPU
    av_GPU = torch.cuda.is_available()
    if  av_GPU == False:
        exit('No available GPU')
    # parse args
    args = args_parser()
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')
	### differential privacy ###

    args.gpu = -1               # -1 (CPU only) or GPU = 0
    args.lr = 0.01          # 0.001 for cifar dataset
    args.model = 'mlp'         # 'mlp' or 'cnn'
    args.dataset = 'mnist'     # 'mnist'
    args.num_users = 10         ### numb of users ###
    args.num_Chosenusers = 10
    args.epochs = 50           # numb of global iters
    args.local_ep = 5        # numb of local iters
    args.num_items_train = 128 # numb of local data size # 
    args.num_items_test =  128
    args.local_bs = 64         ### Local Batch size (1200 = full dataset ###
                               ### size of a user for mnist, 2000 for cifar) ###
                               
    args.set_num_users = [10]
    args.set_local_ep = [5]
    args.set_ratio_realdata = [0.5]
    
    args.num_experiments = 1
    args.clipthr = 10
    args.num_epochs_gener = 50
    args.c_gan = True
    
    args.iid = True
    
    
    #####-Choose Variable-#####
    set_variable = copy.deepcopy(args.set_num_users)
    set_variable0 = copy.deepcopy(args.set_local_ep)
    set_variable1 = copy.deepcopy(args.set_ratio_realdata)
    
    if not os.path.exists('./experiresult'):
        os.mkdir('./experiresult')
          
    # load dataset and split users
    dict_users,dict_users_train,dict_users_test = {},{},{}
    dataset_train,dataset_test = [],[]
    dataset_train = datasets.MNIST('./dataset/', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
    dataset_test = datasets.MNIST('./dataset/', train=False, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
        
    # sample users
    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users, args.num_items_train)
        # dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_items_test) 
        dict_sever = mnist_iid(dataset_test, args.num_users, args.num_items_test)
    else:
        dict_users = mnist_noniid(dataset_train, args.num_users)
        dict_users_test = mnist_noniid(dataset_test, args.num_users)

    img_size = dataset_train[0][0].shape
    
    for v in range(len(set_variable)):
        args.num_users = copy.deepcopy(set_variable[v])
        gener_data,dict_users_gener = generate_sample(args,dataset_train,dict_users)
        final_train_loss = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_train_acc = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_test_loss = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        final_test_acc = [[0 for i in range(len(set_variable1))] for j in range(len(set_variable0))]
        for m in range(args.num_experiments):
            for j in range(len(set_variable1)):
                args.ratio_realdata = copy.deepcopy(set_variable1[j])
                dataset_train_fin, dict_users_fin = data_assign(args,dataset_train,dict_users,gener_data, dict_users_gener)
                for s in range(len(set_variable0)):
                    args.num_Chosenusers = copy.deepcopy(args.num_users)
                    args.local_ep = copy.deepcopy(set_variable0[s])
                    print("dataset:", args.dataset, " num_users:", args.num_users, " num_chosen_users:", args.num_Chosenusers,\
                          " epochs:", args.epochs, "local_ep:", args.local_ep, "local train size", args.num_items_train, "batch size:", args.local_bs)        
                    loss_test, loss_train = [], []
                    acc_test, acc_train = [], []          
                    # build model
                    net_glob = None
                    if args.model == 'cnn' and args.dataset == 'mnist':
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            # net_glob = CNNMnist(args=args).cuda()
                            net_glob = CNN_test(args=args).cuda()
                        else:
                            net_glob = CNNMnist(args=args)
                    elif args.model == 'mlp':
                        len_in = 1
                        for x in img_size:
                            len_in *= x
                        if args.gpu != -1:
                            torch.cuda.set_device(args.gpu)
                            net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).cuda()
                            # net_glob = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes).cuda()
                        else:
                            net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes)
                            # net_glob = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes)
                    else:
                        exit('Error: unrecognized model')
                    print("Nerual Net:",net_glob)
                
                    net_glob.train()  #Train() does not change the weight values
                    # copy weights
                    w_glob = net_glob.state_dict()       
                    w_size = 0
                    w_size_all = 0
                    for k in w_glob.keys():
                        size = w_glob[k].size()
                        if(len(size)==1):
                            nelements = size[0]
                        else:
                            nelements = size[0] * size[1]
                        w_size += nelements*4
                        w_size_all += nelements
                        # print("Size ", k, ": ",nelements*4)
                    print("Weight Size:", w_size, " bytes")
                    print("Weight & Grad Size:", w_size*2, " bytes")
                    print("Each user Training size:", 784* 8/8* args.local_bs, " bytes")
                    print("Total Training size:", 784 * 8 / 8 * 60000, " bytes")
                    # training
                    loss_avg_list, acc_avg_list, list_loss, loss_avg = [], [], [], []  
                    ###  FedAvg Aglorithm  ###
                    ### Compute noise scale ###       
                    for iter in range(args.epochs):
                        print('\n','*' * 20,f'Epoch: {iter}','*' * 20)
                        if  args.num_Chosenusers < args.num_users:
                            chosenUsers = random.sample(range(args.num_users),args.num_Chosenusers)
                            chosenUsers.sort()
                        else:
                            chosenUsers = range(args.num_users)
                        print("\nChosen users:", chosenUsers)                
                        w_locals, w_locals_1ep, loss_locals, acc_locals = [], [], [], []
                        for idx in range(len(chosenUsers)):
                            local = LocalUpdate(args=args, dataset=dataset_train_fin,\
                                        idxs=dict_users_fin[chosenUsers[idx]], tb=summary)
                            w_1st_ep, w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob))
                            w_locals.append(copy.deepcopy(w))
                            ### get 1st ep local weights ###
                            w_locals_1ep.append(copy.deepcopy(w_1st_ep))            
                            loss_locals.append(copy.deepcopy(loss))
                            # print("User ", chosenUsers[idx], " Acc:", acc, " Loss:", loss)
                            acc_locals.append(copy.deepcopy(acc))
                            
                        ### update global weights ###                
                        # w_locals = users_sampling(args, w_locals, chosenUsers)
                        w_glob = average_weights(w_locals) 
                         
                        # copy weight to net_glob
                        net_glob.load_state_dict(w_glob)
                        # global test
                        list_acc, list_loss = [], []
                        net_glob.eval()
                        for c in range(args.num_users):
                            net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=dict_sever[idx], tb=summary)
                            acc, loss = net_local.test(net=net_glob)                    
                            # acc, loss = net_local.test_gen(net=net_glob, idxs=dict_users[c], dataset=dataset_test)
                            list_acc.append(acc)
                            list_loss.append(loss)
                        # print("\nEpoch: {}, Global test loss {}, Global test acc: {:.2f}%".\
                        #      format(iter, sum(list_loss) / len(list_loss),100. * sum(list_acc) / len(list_acc)))
                        # print loss
                        loss_avg = sum(loss_locals) / len(loss_locals)
                        acc_avg = sum(acc_locals) / len(acc_locals)
                        loss_avg_list.append(loss_avg)
                        acc_avg_list.append(acc_avg) 
                        print("\nTrain loss: {}, Train acc: {}".\
                              format(loss_avg_list[-1], acc_avg_list[-1]))
                        print("\nTest loss: {}, Test acc: {}".\
                              format(sum(list_loss) / len(list_loss), sum(list_acc) / len(list_acc)))
                        
                    loss_train=copy.deepcopy(loss_avg)
                    acc_train=copy.deepcopy(acc_avg)
                    loss_test=copy.deepcopy(sum(list_loss) / len(list_loss))
                    acc_test=copy.deepcopy(sum(list_acc) / len(list_acc))
                    
                    perfor_ratio = m/(m+1)
                    final_train_loss[s][j] = copy.deepcopy(perfor_ratio*final_train_loss[s][j]+(1-perfor_ratio)*loss_train)
                    final_train_acc[s][j] = copy.deepcopy(perfor_ratio*final_train_acc[s][j]+(1-perfor_ratio)*acc_train)
                    final_test_loss[s][j] = copy.deepcopy(perfor_ratio*final_test_loss[s][j]+(1-perfor_ratio)*loss_test)
                    final_test_acc[s][j] = copy.deepcopy(perfor_ratio*final_test_acc[s][j]+(1-perfor_ratio)*acc_test)
    
            print('\nFinal train loss:', final_train_loss)
            print('\nFinal train acc:', final_train_acc)
            print('\nFinal test loss:', final_test_loss)
            print('\nFinal test acc:', final_test_acc)
            
            timeslot = int(time.time())
            data_train_loss = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_train_loss)
            data_train_loss.to_csv('./experiresult/'+'train_loss_{}_{}.csv'.format(set_variable[v],timeslot))
            data_test_loss = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_test_loss)
            data_test_loss.to_csv('./experiresult/'+'test_loss_{}_{}.csv'.format(set_variable[v],timeslot))
            data_train_acc = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_train_acc)
            data_train_acc.to_csv('./experiresult/'+'train_acc_{}_{}.csv'.format(set_variable[v],timeslot))
            data_test_acc = pd.DataFrame(index = set_variable0, columns = set_variable1, data = final_test_acc)
            data_test_acc.to_csv('./experiresult/'+'test_acc_{}_{}.csv'.format(set_variable[v],timeslot))