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
    output = np.around(output*10/36)%100
    return input, output

def suffle_data():
    file_list = np.array(files)
    shuffle = np.random.permutation(filenum)

np.resize
w = Wind()
w.load_state_dict(tt.load('./wind_modul.pth'))
for p in w.parameters():
    print(p)
#res = models.resnet18(pretrained=True)
#print(res)
#optim = tt.optim.Adam(w.parameters(), lr=1e-4)
#critization = nn.BCELoss()
files = os.listdir('../data/dataset_ascending/')
files = np.array(files)
file_num = len(files)
batch_size = 32
shuffle = np.random.permutation(file_num)
y_ = np.array([])
out_ = np.array([])
for j in range(file_num//32):
    print(j)
    input, label = get_data(files[shuffle[j*32:j*32+32]])
    y_ = np.concatenate((y_, label))
    label = tt.LongTensor(label.reshape(32,1))
    x = Variable(tt.FloatTensor(input), volatile=True)
    label = tt.zeros(batch_size, 100).scatter_(1, label, 1)
    y = Variable(label, volatile=True)
    out = w(x)
    out_ = np.concatenate((out_, np.argmax(out.data.numpy(), 1)))
    del input, label
    gc.collect()

y_ = y_*3.6
print(y_)
out_ = out_*3.6
print(out_)
print('rms')
rms=np.sqrt(np.sum((y_-out_)**2)/len(y_))
print(rms)
import matplotlib.pyplot as plt
plt.scatter(y_, out_,)
plt.text(350, 350, 'rms=%f'%rms, fontsize=15)
plt.savefig('linear.png')
