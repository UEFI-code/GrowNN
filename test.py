import torch
import cv2
import torch
from torch.utils.data import DataLoader
from torch import nn ,optim
from torch.autograd import Variable
from config import config
from datasets import *
import Model
from utils.utils import accuracy
classes= {0:"Surprise",1:"Neture",2:"Happy",3:"Angry"}
import os
def evaluate(test_loader,model,criterion):
    sum = 0
    test_loss_sum = 0
    test_top1_sum = 0
    model.eval()

    for ims, label in test_loader:
        input_test = Variable(ims).cuda()
        target_test = Variable(torch.from_numpy(np.array(label)).long()).cuda()
        output_test, _, _, _ = model(input_test)
        loss = criterion(output_test, target_test)
        top1_test = accuracy(output_test, target_test, topk=(1,))
        sum += 1
        test_loss_sum += loss.data.cpu().numpy()
        test_top1_sum += top1_test[0].cpu().numpy()[0]
    avg_loss = test_loss_sum / sum
    avg_top1 = test_top1_sum / sum
    return avg_loss, avg_top1


def test(test_loader,model):
    model.eval()
    predict_file = open("%s.txt" % config.model_name, 'w')
    for i, (input,filename) in enumerate(tqdm(test_loader)):
        y_pred = model(input)
        smax = nn.Softmax(1)
        smax_out = smax(y_pred)
        pred_label = np.argmax(smax_out.cpu().data.numpy())
        predict_file.write(filename[0]+', ' +classes[pred_label]+'\n')

def test_one_image(image,model,index):
    model.eval()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (96,96))
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)
    img = Variable(img)
    y_pred = model(img)
    smax = nn.Softmax()
    smax_out = smax(y_pred)
    pred_label = np.argmax(smax_out.cpu().data.numpy())
 
