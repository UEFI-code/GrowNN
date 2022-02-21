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
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    predict_file = open("%s.txt" % config.model_name, 'w')
    for i, (input,filename) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            input = Variable(input).cuda()
        else:
            input= Variable(input)
        #print("input.size = ",input.data.shape)
        y_pred = model(input)
        smax = nn.Softmax(1)
        smax_out = smax(y_pred)
        pred_label = np.argmax(smax_out.cpu().data.numpy())
        predict_file.write(filename[0]+', ' +classes[pred_label]+'\n')


if __name__ == '__main__':

    test_list, _ = get_files(config.data_folder,config.ratio)
    test_loader = DataLoader(datasets(test_list, transform=None,test = True), batch_size= 1, shuffle=False,collate_fn=collate_fn, num_workers=4)
    model = Model.myModel()
    checkpoint = torch.load(config.weights+ config.model_name+'.pth')
    model.load_state_dict(checkpoint["state_dict"])
    index = 0
    #optimizer.load_state_dict(checkpoint["optimizer"])
    print("Start Test.......")
    for file in os.listdir('../DataSet/train/l1/Neture'):
        f = '../DataSet/train/l1/Neture'+ '/' + file
    #test(test_loader,model)
        image = cv2.imread(f)
        sname = 'Neture-%d' % index
        test_one_image(image,model,sname)
        index = index + 1

