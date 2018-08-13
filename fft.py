import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import signal

def get_data(n):
    files = os.listdir('../data/dataset_ascending/')
    if n > len(files):
        n = len(files)
    os.chdir('../data/dataset_ascending/')
    vv = []
    vh = []
    incidence = []
    direction = []
    speed = []
    height = []
    for file in files[:n]:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            sigma_vv = data['Sigma0_VV_db:float']
            vv.append(np.resize(sigma_vv, (1, 160000)).reshape(400, 400))
            sigma_vh = data['Sigma0_VH_db:float']
            vh.append(np.resize(sigma_vh, (1, 160000)).reshape(400,400))
            speed.append(data['wspd'])
            height.append(data['height'])
            direction.append(data['wdir'])
            incidence.append(data['incident_angle:float'])
    ten_wind = [float(x)*pow(10/float(y), 0.11) for x, y in zip(speed, height)]
    return vv, vh, ten_wind, direction, incidence

def open_text(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        sigma_vv = data['Sigma0_VV:float']
        sigma_vh = data['Sigma0_VH:float']
        incidence = data['incident_angle:float']
    return sigma_vv, sigma_vh, incidence

def sobel(img):
    r = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])/32
    d = r+r.T*1j
    sob1 = signal.convolve2d(img, r,mode='valid')
    sob2 = signal.convolve2d(img, r.T,mode='valid')
    return sob1, sob2


if __name__ == '__main__':
#    sigma_vv, sigma_vh, w_speed, w_direction, incidence = get_data(10)
#    plt.imshow(sigma_vv[0])
#    print(w_speed[0])
#    print(w_direction[0])
#    plt.show()
    #img = plt.imread('./sierra.jpg')
    sigma_vv, sigma_vh, incidence = open_text('subset_0_of_S1B_IW_GRDH_1SDV_20170507T233247_20170507T233312_005499_009A2C_4206_Noise-Cor_Cal_Spk.csv.txt')
    sigma_vv = np.reshape(sigma_vv, (1241,1479))
    sigma_vv = sigma_vv[0:1000,:1000]
    sigma_vv = 10*np.log10(sigma_vv)
    print(sigma_vv)
    fft = np.fft.fft2(sigma_vv)
    fftc = np.fft.fftshift(fft)
    plt.subplot(1, 2, 1)
    plt.imshow(sigma_vv,'Greys')
    plt.subplot(1, 2, 2)

    img = np.log(np.abs(fftc))
    plt.imshow(img, 'Greys')
    plt.colorbar()
    plt.show()
