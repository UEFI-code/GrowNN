import numpy as np
import torch
import torchvision
import os
from config import config
import Model
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import *
from test import *
from utils.utils import*

def getModelPara(model):
    params = list(model.layera.parameters()) + list(model.layerb.parameters()) + list(model.layerc.parameters())
    for i in model.li0_group:
        params += list(i.parameters())
    for i in model.li1_group:
        params += list(i.parameters())
    for i in model.li2_group:
        params += list(i.parameters())
    params += list(model.li3.parameters())
    return params

if __name__ == '__main__' :
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    model = Model.myModel()
    if torch.cuda.is_available():
        model = model.cuda()
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(getModelPara(model), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    optimizer.zero_grad()
    start_epoch = 0
    current_accuracy = 0
    resume = False
    if resume:
        checkpoint = torch.load(config.weights+ config.model_name+'.pth')
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    transform = transforms.Compose([
                                    transforms.RandomResizedCrop(90),
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    transforms.RandomRotation(30),
                                    transforms.RandomGrayscale(p = 0.5),
                                    transforms.Resize((config.img_width, config.img_height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    _, train_list = get_files(config.data_folder,config.ratio)
    input_data = datasets(train_list,transform= transform)
    train_loader = DataLoader(input_data,batch_size = config.batch_size,shuffle = True,collate_fn = collate_fn ,pin_memory=False,num_workers=4)

    test_list, _ = get_files(config.data_folder, config.ratio)
    test_loader = DataLoader(datasets(test_list,transform = None),batch_size= config.batch_size,shuffle =False,collate_fn = collate_fn,num_workers=4)

    train_loss = []
    acc = []
    test_loss = []
    totalAddCells = 0
    #model.li0_group.append(nn.Linear(128,128))
    #new_weight = model.li0_group[len(model.li0_group)-1].weight * 0
    #new_bias = model.li0_group[len(model.li0_group)-1].bias * 0
    #model.li0_group[len(model.li0_group)-1].weight = nn.Parameter(new_weight)
    #model.li0_group[len(model.li0_group)-1].bias = nn.Parameter(new_bias)
    print("------ Start Training ------\n")
    for epoch in range(start_epoch,config.epochs):
        model.train()
        config.lr = lr_step(epoch)
        loss_epoch = 0
        for index,(input,target) in enumerate(train_loader):
            need_rst_optim = False
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output, li0_dat, li1_dat, li2_dat = model(input)
            loss = criterion(output,target)
            try: 
                loss.backward()
            except:
                print('backward failed 233')
            optimizer.step()
            loss_epoch += loss.item()
            #Check li0!
            for i in range(len(li0_dat)) :
                for j in range(len(li0_dat[i][0])):
                    if li0_dat[i][0][j] > 0.7 or li0_dat[i][0][j] < 0 - 0.7 :
                        #Found this Nerve so tired
                        need_rst_optim = True
                        totalAddCells += 1
                        exp_weight = model.li0_group[i].weight.tolist()
                        exp_bias = model.li0_group[i].bias.tolist()
                        for k in range(len(exp_weight[j])):
                            exp_weight[j][k] /= 2
                        exp_bias[j] /= 2
                        model.li0_group[i].weight = nn.Parameter(torch.tensor(exp_weight))
                        model.li0_group[i].bias = nn.Parameter(torch.tensor(exp_bias))
                        #Seek for unborned Cell
                        foundCell = False
                        for a in range(len(model.li0_group)):
                            for b in range(len(model.li0_group[a].weight)):
                                if model.li0_group[a].weight[b].sum() < 0.01:
                                    print('Found a unBorned Cell for li0')
                                    bak_weight = model.li0_group[a].weight.tolist()
                                    bak_bias = model.li0_group[a].bias.tolist()
                                    for c in range(len(bak_weight[b])):
                                        bak_weight[b][c] = exp_weight[j][c]
                                    bak_bias[b] = exp_bias[j]
                                    model.li0_group[a].weight = nn.Parameter(torch.tensor(bak_weight))
                                    model.li0_group[a].bias = nn.Parameter(torch.tensor(bak_bias))
                                    foundCell = True
                                    break
                            if foundCell:
                                break
                        if foundCell == False:
                            print('emmm not found unBorn Cell for li0')
                            model.li0_group.append(nn.Linear(128,128))
                            new_weight = model.li0_group[len(model.li0_group)-1].weight * 0
                            new_bias = model.li0_group[len(model.li0_group)-1].bias * 0
                            new_weight = new_weight.tolist()
                            new_bias = new_bias.tolist()
                            for k in range(len(new_weight[j])): 
                                new_weight[j][k] = exp_weight[j][k]
                            new_bias[j] = exp_bias[j]
                            model.li0_group[len(model.li0_group)-1].weight = nn.Parameter(torch.tensor(new_weight))
                            model.li0_group[len(model.li0_group)-1].bias = nn.Parameter(torch.tensor(new_bias))
                            model.li0_group[len(model.li0_group)-1].zero_grad()
                            print('Append to Linear 0, now %d layers' % len(model.li0_group))

            #Check li1!
            for i in range(len(li1_dat)) :
                for j in range(len(li1_dat[i][0])):
                    if li1_dat[i][0][j] > 0.7 or li1_dat[i][0][j] < 0 - 0.7 :
                        #Found this Nerve so tired
                        need_rst_optim = True
                        totalAddCells += 1
                        exp_weight = model.li1_group[i].weight.tolist()
                        exp_bias = model.li1_group[i].bias.tolist()
                        for k in range(len(exp_weight[j])):
                            exp_weight[j][k] /= 2
                        exp_bias[j] /= 2
                        model.li1_group[i].weight = nn.Parameter(torch.tensor(exp_weight))
                        model.li1_group[i].bias = nn.Parameter(torch.tensor(exp_bias))
                        #Seek for unborned Cell
                        foundCell = False
                        for a in range(len(model.li1_group)):
                            for b in range(len(model.li1_group[a].weight)):
                                if model.li1_group[a].weight[b].sum() < 0.01:
                                    print('Found a unBorned Cell for li1')
                                    bak_weight = model.li1_group[a].weight.tolist()
                                    bak_bias = model.li1_group[a].bias.tolist()
                                    for c in range(len(bak_weight[b])):
                                        bak_weight[b][c] = exp_weight[j][c]
                                    bak_bias[b] = exp_bias[j]
                                    model.li1_group[a].weight = nn.Parameter(torch.tensor(bak_weight))
                                    model.li1_group[a].bias = nn.Parameter(torch.tensor(bak_bias))
                                    foundCell = True
                                    break
                            if foundCell:
                                break
                        if foundCell == False:
                            print('emmm not found unBorn Cell for li1')
                            model.li1_group.append(nn.Linear(128,128))
                            new_weight = model.li1_group[len(model.li1_group)-1].weight * 0
                            new_bias = model.li1_group[len(model.li1_group)-1].bias * 0
                            new_weight = new_weight.tolist()
                            new_bias = new_bias.tolist()
                            for k in range(len(new_weight[j])):
                                new_weight[j][k] = exp_weight[j][k]
                            new_bias[j] = exp_bias[j]
                            model.li1_group[len(model.li1_group)-1].weight = nn.Parameter(torch.tensor(new_weight))
                            model.li1_group[len(model.li1_group)-1].bias = nn.Parameter(torch.tensor(new_bias))
                            model.li1_group[len(model.li1_group)-1].zero_grad()
                            print('Append to Linear 1, now %d layers' % len(model.li1_group))
            #Check li2!
            for i in range(len(li2_dat)) :
                for j in range(len(li2_dat[i][0])):
                    if li2_dat[i][0][j] > 0.7 or li2_dat[i][0][j] < 0 - 0.7 :
                        #Found this Nerve so tired
                        need_rst_optim = True
                        totalAddCells += 1
                        exp_weight = model.li2_group[i].weight.tolist()
                        exp_bias = model.li2_group[i].bias.tolist()
                        for k in range(len(exp_weight[j])):
                            exp_weight[j][k] /= 2
                        exp_bias[j] /= 2
                        model.li2_group[i].weight = nn.Parameter(torch.tensor(exp_weight))
                        model.li2_group[i].bias = nn.Parameter(torch.tensor(exp_bias))
                        #Seek for unborned Cell
                        foundCell = False
                        for a in range(len(model.li2_group)):
                            for b in range(len(model.li2_group[a].weight)):
                                if model.li2_group[a].weight[b].sum() < 0.01:
                                    print('Found a unBorned Cell for li2')
                                    bak_weight = model.li2_group[a].weight.tolist()
                                    bak_bias = model.li2_group[a].bias.tolist()
                                    for c in range(len(bak_weight[b])):
                                        bak_weight[b][c] = exp_weight[j][c]
                                    bak_bias[b] = exp_bias[j]
                                    model.li2_group[a].weight = nn.Parameter(torch.tensor(bak_weight))
                                    model.li2_group[a].bias = nn.Parameter(torch.tensor(bak_bias))
                                    foundCell = True
                                    break
                            if foundCell:
                                break
                        if foundCell == False:
                            print('emmm not found unBorn Cell for li2')
                            model.li2_group.append(nn.Linear(128,128))
                            new_weight = model.li2_group[len(model.li2_group)-1].weight * 0
                            new_bias = model.li2_group[len(model.li2_group)-1].bias * 0
                            new_weight = new_weight.tolist()
                            new_bias = new_bias.tolist()
                            for k in range(len(new_weight[j])):
                                new_weight[j][k] = exp_weight[j][k]
                            new_bias[j] = exp_bias[j]
                            model.li2_group[len(model.li2_group)-1].weight = nn.Parameter(torch.tensor(new_weight))
                            model.li2_group[len(model.li2_group)-1].bias = nn.Parameter(torch.tensor(new_bias))
                            model.li2_group[len(model.li2_group)-1].zero_grad()
                            print('Append to Linear 2, now %d layers' % len(model.li2_group))
            if need_rst_optim:
                optimizer = optim.SGD(getModelPara(model), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
                optimizer.zero_grad()
            if (index+1) % 10 == 0:
                print("Epoch: {} [{:>3d}/{}]\t Loss: {:.6f} ".format(epoch+1,index*config.batch_size,len(train_loader.dataset),loss.item()))
                print("Total added " + str(totalAddCells) + " cells")
        if (epoch+1) % 1 ==0:
            print("\n------ Evaluate ------")
            model.eval()
            # evaluate the model on the test data
            test_loss1, accTop1 = evaluate(test_loader,model,criterion)
            acc.append(accTop1)
            print("type(accTop1) =",type(accTop1))
            test_loss.append(test_loss1)
            train_loss.append(loss_epoch/len(train_loader))
            print("Test_epoch: {} Test_accuracy: {:.4}% Test_Loss: {:.6f}".format(epoch+1,accTop1,test_loss1))
            save_model = accTop1 > current_accuracy
            accTop1 = max(current_accuracy,accTop1)
            current_accuracy = accTop1
            print("Best Accu = %f" % current_accuracy)
            save_checkpoint({
                "epoch": epoch + 1,
                "model_name": config.model_name,
                "state_dict": model.state_dict(),
                "accTop1": current_accuracy,
                "optimizer": optimizer.state_dict(),
            }, save_model)
    open('finish.txt','w').write(str(accTop1/100))
