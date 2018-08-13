import torch as tt
from torch.autograd import Variable
import torch.nn as nn
import torchvision as tv
import  torchvision.models  as models
import pickle
import os
import gc
import numpy as np
import tqdm
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
            nn.Sigmoid()
        )

    def forward(self, input):
        return tt.squeeze(self.main(input))




def get_data(l):
    input = []
    output = []
    for file in l:
        with open('/Volumes/Temperament/data/dataset_ascending/'+file, 'rb') as f:
            data = pickle.load(f)
            x1 = np.reshape(np.resize(data['Sigma0_VH_db:float'], (1, 160000)), (400, 400))

            x2 = np.reshape(np.resize(data['incident_angle:float'], (1, 160000)), (400, 400))
            x3 = np.reshape(np.resize(data['Sigma0_VV_db:float'], (1, 160000)), (400, 400))
            input.append([x3, x1, x2])
            del data
            gc.collect()
        output.append(float(file.split('_')[4]))
    #output = np.around(output*10/36)%100
    return input, output
w = Wind()
w.load_state_dict(tt.load('./res_wind_modul.pth'))
files = np.array(os.listdir('/Volumes/Temperament/data/dataset_ascending/'))
out = []
out_put = []
for i in tqdm.tqdm(range(len(files)//32)):
    input, label = get_data(files[i*32:(i+1)*32])
    out.extend(label)
    x = Variable(tt.FloatTensor(input), requires_grad=False)
    out1 = w(x)
    out_ = np.argmax(out1.data.numpy(), 1)
    y_ = out_*3.6
    out_put.extend(y_)

plt.plot(out, out_put, 'o')
plt.xlabel('buoy')
plt.ylabel('nn')

plt.savefig('nn')
plt.show()


