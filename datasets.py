import torch
from torch.utils.data import Dataset,DataLoader
import cv2
import os
from tqdm import tqdm
from config import config
from glob import glob
import os
from torchvision import transforms
import numpy as np
import random
from shutil import copy
from PIL import Image
import math

def get_files(file_dir,ratio):
    Surprise = []
    labels_Surprise = []
    Neture = []
    labels_Neture = []
    Happy = []
    labels_Happy=[]
    Anger = []
    labels_Anger = []
    for file  in os.listdir(file_dir +'Surprise'):
        Surprise.append(file_dir + 'Surprise' + '/' + file)
        labels_Surprise.append(0)
    for file in os.listdir(file_dir + 'Neture'):
        Neture.append(file_dir + 'Neture' + '/' + file)
        labels_Neture.append(1)
    for file in os.listdir(file_dir + 'Happy'):
        Neture.append(file_dir + 'Happy' + '/' +file)
        labels_Happy.append(2)
    for file in os.listdir(file_dir + 'Anger'):
        Anger.append(file_dir + 'Anger' + '/' +file)
        labels_Anger.append(3)
    image_list = np.hstack((Surprise ,Neture, Happy, Anger))
    labels_list = np.hstack((labels_Surprise, labels_Neture, labels_Happy, labels_Anger))
    temp = np.array([image_list, labels_list]).transpose()
    np.random.shuffle(temp)
    n_test = int(math.ceil(len(temp) * ratio))
    test_data = temp[0:n_test,:]
    train_data = temp[n_test:-1,:]
    return test_data,train_data

class datasets(Dataset):
    def __init__(self,data,transform = None,test = False):
        imgs = []
        labels = []
        self.test = test
        self.data = data.transpose()
        self.len = len(data)
        self.transform = transform
        self.imgs = self.data[0]
        self.labels = self.data[1]
    def __getitem__(self,index):
        if self.test:
            filename = self.imgs[index]
            filename = filename
            img_path = self.imgs[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (config.img_width, config.img_height))
            img = transforms.ToTensor()(img)
            return img,filename
        else:
            img_path = self.imgs[index]
            label = self.labels[index]
            label = int(label)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(config.img_width,config.img_height))
            # img = transforms.ToTensor()(img)

            if self.transform is not None:
                img = Image.fromarray(img)
                img = self.transform(img)

            else:
                img = transforms.ToTensor()(img)
            return img,label

    def __len__(self):
        return self.len

def collate_fn(batch):
    imgs = []
    label = []
    for i in batch:
      imgs.append(i[0])
      label.append(i[1])
    return torch.stack(imgs, 0),label

if __name__ == '__main__':
    test_data,_ = get_files(config.data_folder,0.2)
    for i in (test_data):
        print(i)
    print(len(test_data))

    transform = transforms.Compose([transforms.ToTensor()])
    data = datasets(test_data,transform = transform)
    #print(data[0])
