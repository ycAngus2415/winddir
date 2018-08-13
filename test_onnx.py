import torch as tt
from torch.autograd import Variable
import torch.nn as nn
import torchvision as tv
import  torchvision.models  as models
import pickle
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch.onnx

class Wind(nn.Module):
    def __init__(self):
        super(Wind, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, (13, 13), 3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64,(7, 7), 3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,(6, 6), 3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,(3, 3), 3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,100 ,(4, 4), 3, bias=False),
            nn.ReLU(),
            nn.Softmax()
        )

    def forward(self, input):
        return tt.squeeze(self.main(input))

#class Wind(nn.Module):
#    def __init__(self):
#        super(Wind, self).__init__()
#        self.cov1 = nn.Conv2d(3, 32, (13, 13), 3, bias=False)
#        self.cov2 = nn.Conv2d(32, 64,(7, 7), 3, bias=False)
#        self.cov3 = nn.Conv2d(64,128,(6, 6), 3, bias=False)
#        self.cov4 = nn.Conv2d(128,256,(3, 3), 3, bias=False)
#        self.cov5 = nn.Conv2d(256,100 ,(4, 4), 3, bias=False)
#        self.relu = nn.ReLU()
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, input):
#        return tt.squeeze(self.sigmoid(self.cov5(self.relu(self.cov4(self.relu(self.cov3(self.relu(self.cov2(self.relu(self.cov1(input)))))))))))


# l is files list
def get_data(l):
    input = []
    output = []
    for file in l:
        with open('../data/dataset_ascending/'+file, 'rb') as f:
            data = pickle.load(f)
            x1 = np.reshape(np.resize(data['Sigma0_VH_db:float'], (1, 160000)), (400, 400))
            x2 = np.reshape(np.resize(data['incident_angle:float'], (1, 160000)), (400, 400))
            x3 = np.reshape(np.resize(data['Sigma0_VV_db:float'], (1, 160000)), (400, 400))
            input.append([x3, x1, x2])
            del data
            gc.collect()
        output.append(float(file.split('_')[4]))
    output = np.array(output)
    output = np.around(output*100/36)%1000
    return input, output

#w = Wind()
w = models.resnet18(pretrained=True)
print(w)
optim = tt.optim.Adam(w.parameters(), lr=1e-4)
critization = nn.BCELoss()
files = os.listdir('../data/dataset_ascending/')
files = np.array(files)
file_num = len(files)
batch_size = 32
for i in range(20):
    shuffle = np.random.permutation(file_num)
    for j in range(file_num//32):
        print(i, end=', ')
        print(j, end=', ')
        print('loss: ', end ='')
        input, label = get_data(files[shuffle[j*32:j*32+32]])
        label = tt.LongTensor(label.reshape(32,1))
        x = Variable(tt.FloatTensor(input), requires_grad=True)
        label = tt.zeros(batch_size, 1000).scatter_(1, label, 1)
        y = Variable(label)
        torch.onnx.export(w, x, 'squee.onnx', verbose=True)
        break
        out1 = w(x)
        soft = nn.Softmax()
        out = soft(out1)
        loss = critization(out, y)
        loss.backward()
        optim.step()
        print(loss.data.numpy()[0])
    break
    input, label = get_data(files[shuffle[-(file_num%32):]])
    x = Variable(tt.FloatTensor(input), volatile=True)
    out = w(x)
    out_ = np.argmax(out.data.numpy(), 1)
    y_ = label*0.36
    out_ = out_*0.36
    rms = np.sqrt(np.sum((y_-out_)**2)/len(y_))
    print(i, end=' ')
    print('rms: ', end=' ' )
    print(rms)
    del input, label, rms
    gc.collect()


for p in w.parameters():
    print(p)
tt.save(w.state_dict(), 'res_wind_modul.pth')
tt.save(optim.state_dict(), 'res_optim.pth')
