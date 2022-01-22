# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:19:31 2019

@author: WEIKANG
"""
import torch
import numpy as np
import copy
import math

def f_zero(args, f, num_iter):
    x0 = 0
    x1 = args.max_epochs
    if f(x0)*f(x1)>=0:
        if abs(f(x0))>abs(f(x1)):
            x0 = copy.deepcopy(x1)
    else:
        y = copy.deepcopy(args.max_epochs)
        for i in range(100):
            if f(x0)*f(x1)<0:
                y = copy.deepcopy(x1)
                x1 = copy.deepcopy((x0+x1)/2)
            else:
                x1 = copy.deepcopy(y)
                x0 = copy.deepcopy((x0+x1)/2)
            if abs(x0-x1)<0.01:
                break
        if (x0+num_iter) > args.max_epochs:
            x0 = copy.deepcopy(args.max_epochs)    
    return x0

def get_l2_norm(args, params_a):
    sum = 0
    if args.gpu != -1:
        tmp_a = np.array([v.detach().cpu().numpy() for v in params_a])
    else:
        tmp_a = np.array([v.detach().numpy() for v in params_a])
    a = []
    for i in tmp_a:
        x = i.flatten()
        for k in x:
            a.append(k)
    for i in range(len(a)):
        sum += (a[i] - 0) ** 2
    norm = np.sqrt(sum)
    return norm

def get_1_norm(params_a):
    sum = 0
    if isinstance(params_a,np.ndarray) == True:
        sum += pow(np.linalg.norm(params_a, ord=2),2) 
    else:
        for i in params_a.keys():
            if len(params_a[i]) == 1:
                sum += pow(np.linalg.norm(params_a[i].cpu().numpy(), ord=2),2)
            else:
                a = copy.deepcopy(params_a[i].cpu().numpy())
                for j in a:
                    x = copy.deepcopy(j.flatten())
                    sum += pow(np.linalg.norm(x, ord=2),2)                  
    norm = np.sqrt(sum)
    return norm

def get_2_norm(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        if len(params_a[i]) == 1:
            sum += pow(np.linalg.norm(params_a[i].cpu().numpy()-\
                params_b[i].cpu().numpy(), ord=2),2)
        else:
            a = copy.deepcopy(params_a[i].cpu().numpy())
            b = copy.deepcopy(params_b[i].cpu().numpy())
            x = []
            y = []
            for j in a:
                x.append(copy.deepcopy(j.flatten()))
            for k in b:          
                y.append(copy.deepcopy(k.flatten()))
            for m in range(len(x)):
                sum += pow(np.linalg.norm(x[m]-y[m], ord=2),2)            
    norm = np.sqrt(sum)
    return norm

def inner_product(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        sum += np.sum(np.multiply(params_a[i].cpu().numpy(),\
                params_b[i].cpu().numpy()))     
    return sum

def avg_grads(g):
    grad_avg = copy.deepcopy(g[0])
    for k in grad_avg.keys():
        for i in range(1, len(g)):
            grad_avg[k] += g[i][k]
        grad_avg[k] = torch.div(grad_avg[k], len(g))
    return grad_avg