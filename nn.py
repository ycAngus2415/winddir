import torch as tt
from torch.autograd import Variable
import torch.nn as nn
import torchvision as tv
import  torchvision.models  as models
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

res = models.resnet18(pretrained=True)
print(res)
files = os.listdir('../data/dataset_ascending/')
output = []
for file in files:
    with open('../data/dataset_ascending/speckle_cal_41043_subset_46.0_ 4.3_4_S1A_IW_GRDH_1SDV_20170526T222855_20170526T222920_016759_01BD7F_85C4.txt', 'rb') as f:
        data = pickle.load(f)
        print(data.keys())
    output.append(file.split('_')[4])
    break

x1 = np.reshape(data['Sigma0_VH_db:float'], (400, 400)
               )
x2 = np.reshape(data['incident_angle:float'], (400, 400))
x3 = np.reshape(data['Sigma0_VV_db:float'], (400, 400))
x = np.array([x3, x1, x2])
input = Variable(tt.FloatTensor(x))
print(input.size())
print(output)
out = res(input.unsqueeze(0))
print(out.data.numpy())
plt.figure(figsize=(100,100))
plt.plot(range(1000), out.squeeze().data.numpy(),'o')
plt.show()
