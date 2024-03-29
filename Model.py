import numpy as np
import torch
import torchvision
from torch import nn,optim
from config import config

class myModel(nn.Module):
    def __init__(self):
        super(myModel,self).__init__()
        self.layera = nn.Sequential(
            nn.Conv2d(3,16,3), #in_channels out_channels kernel_size
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 2,stride = 2)  #149
        )
        self.layerb = nn.Sequential(
            nn.Conv2d(16,32,3,2),  #74    #
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size =2,stride=2)  #37

        )
        self.layerc = nn.Sequential(
            nn.Conv2d(32,32,3,2),  #18
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 2, stride = 2)  #9
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax()
        self.activate = self.relu
        self.li0_group = []
        self.li1_group = []
        self.li2_group = []
        self.li0_group.append(nn.Linear(128,128))
        self.li1_group.append(nn.Linear(128,128))
        self.li2_group.append(nn.Linear(128,128))
        self.li3 = nn.Linear(128,4)
        self.quota_li0 = torch.ones(config.batch_size, 128).cuda() * 10
        self.consume_li0 = torch.zeros(config.batch_size, 128).cuda()
        self.resume_li0 = torch.ones(config.batch_size, 128).cuda() * 1.2
        self.quota_li1 = torch.ones(config.batch_size, 128).cuda() * 10
        self.consume_li1 = torch.zeros(config.batch_size, 128).cuda()
        self.resume_li1 = torch.ones(config.batch_size, 128).cuda() * 1.2
        self.quota_li2 = torch.ones(config.batch_size, 128).cuda() * 10
        self.consume_li2 = torch.zeros(config.batch_size, 128).cuda()
        self.resume_li2 = torch.ones(config.batch_size, 128).cuda() * 1.2
        '''
        #init
        exp_weight = self.li0_group[0].weight.tolist()
        exp_bias = self.li0_group[0].bias.tolist()
        for i in range(int(len(exp_weight)*0.9)):
            for j in range(len(exp_weight[i])):
                exp_weight[i][j] = 0.0
            exp_bias[i] = 0.0
        self.li0_group[0].weight = nn.Parameter(torch.tensor(exp_weight))
        self.li0_group[0].bias = nn.Parameter(torch.tensor(exp_bias))
        #init
        exp_weight = self.li1_group[0].weight.tolist()
        exp_bias = self.li1_group[0].bias.tolist()
        for i in range(int(len(exp_weight)*0.9)):
            for j in range(len(exp_weight[i])):
                exp_weight[i][j] = 0.0
            exp_bias[i] = 0.0
        self.li1_group[0].weight = nn.Parameter(torch.tensor(exp_weight))
        self.li1_group[0].bias = nn.Parameter(torch.tensor(exp_bias))
        #init
        exp_weight = self.li2_group[0].weight.tolist()
        exp_bias = self.li2_group[0].bias.tolist()
        for i in range(int(len(exp_weight)*0.9)):
            for j in range(len(exp_weight[i])):
                exp_weight[i][j] = 0.0
            exp_bias[i] = 0.0
        self.li2_group[0].weight = nn.Parameter(torch.tensor(exp_weight))
        self.li2_group[0].bias = nn.Parameter(torch.tensor(exp_bias))
        '''
    def forward(self, x):
        self.sigmoid.zero_grad()
        x = self.layera(x)
        x = self.layerb(x)
        x = self.layerc(x)
        x = x.view(x.size(0),-1)
        li0_dat = []
        li1_dat = []
        li2_dat = []
        li3_dat = []
        #Ready to 0th Linear
        for i in range(len(self.li0_group)) :
            li0_dat.append(self.li0_group[i].cuda()(x))
            li0_dat[i] = self.activate(li0_dat[i])
            if i == 0 :
                y = li0_dat[i]
            else :
                y = y + li0_dat[i]
        try:
            x = self.li0Tiring(y)
        except:
            x = y
        #Ready to 1th Linear
        for i in range(len(self.li1_group)) :
            li1_dat.append(self.li1_group[i].cuda()(x))
            li1_dat[i] = self.activate(li1_dat[i])
            if i == 0 :
                y = li1_dat[i]
            else :
                y = y + li1_dat[i]
        try:
            x = self.li1Tiring(y)
        except:
            x = y
        #Ready to 2th Linear
        for i in range(len(self.li2_group)) :
            li2_dat.append(self.li2_group[i].cuda()(x))
            li2_dat[i] = self.activate(li2_dat[i])
            if i == 0 :
                y = li2_dat[i]
            else :
                y = y + li2_dat[i]
        try:
            x = self.li2Tiring(y)
        except:
            x = y
        x = self.li3(x)
        x = self.softmax(x)
        return x, li0_dat, li1_dat, li2_dat
    
    def li0Tiring(self, x):
        remain = self.quota_li0 - self.consume_li0
        activate_ratio = self.relu(remain / self.quota_li0)
        x = x * activate_ratio
        self.consume_li0 += x - self.resume_li0
        self.consume_li0 = self.relu(self.consume_li0).detach()
        return x

    def li1Tiring(self, x):
        remain = self.quota_li1 - self.consume_li1
        activate_ratio = self.relu(remain / self.quota_li1)
        x = x * activate_ratio
        self.consume_li1 += x - self.resume_li1
        self.consume_li1 = self.relu(self.consume_li1).detach()
        return x

    def li2Tiring(self, x):
        remain = self.quota_li2 - self.consume_li2
        activate_ratio = self.relu(remain / self.quota_li2)
        x = x * activate_ratio
        self.consume_li2 += x - self.resume_li2
        self.consume_li2 = self.relu(self.consume_li2).detach()
        return x
