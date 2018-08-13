import numpy as np
import h5py
import os
import pickle
import pandas as pd

file = 'subset_0_of_S1B_IW_GRDH_1SDV_20170507T233247_20170507T233312_005499_009A2C_4206_Noise-Cor_Cal_Spk.csv'
dataset = {}
with open(file) as datas:
    data = datas.readlines()
    name = data[1][:-1].split('\t')[1:]
    print(name)
    for n in name:
        dataset[n] = []
    for s in data[2:]:
        li = s[:-1].split('\t')
        for i in range(len(name)):
            dataset[name[i]].append(float(li[i+1]))
    del data
    del name
    with open(file+'.txt', 'wb') as txt:
        pickle.dump(dataset, txt)
    del dataset
