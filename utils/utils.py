#This file include some functions
from config import config
import torch
import os
import shutil

def save_checkpoint(state, save_model):
    filename = config.weights + config.model_name + ".pth"
    torch.save(state, filename)
    if save_model:
        message = config.weights + config.model_name+ '.pth'
        print("Get Better top1 : %s saving weights to %s"%(state["accTop1"],message))
        with open("./logs/%s.txt"%config.model_name,"a") as f:
            print("Get Better top1 : %s saving weights to %s"%(state["accTop1"],message),file=f)


def accuracy(output,target,topk = (1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res =[]
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0,keepdim =True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def lr_step(epoch):
    if epoch < 30:
        lr = 0.01
    elif epoch < 80:
        lr = 0.001
    elif epoch < 120:
        lr = 0.0005
    else:
        lr = 0.0001
    return lr
