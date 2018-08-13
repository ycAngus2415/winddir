import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import signal
import smoothing

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

def sobel_inverse(img):
    r = np.array([[-3,0,3],[-10,0,10],[-3,0,3]])/32
    d = r+r.T*1j
    sob1 = signal.convolve2d(img, r,mode='valid')
    sob2 = signal.convolve2d(img, r.T,mode='valid')
    return sob1, sob2

def main_direction(sob1, sob2):
    g = sob1+sob2*1j
    g1 = smoothing.smooth(g)
    g1 = np.abs(g1)
    g2 = smoothing.smooth(np.abs(g))
    r = g1/g2
    g = g/np.abs(g)
    g = g.ravel()
    x = g.real
    y = g.imag
    x1 = np.copy(x)
    y1 = np.copy(y)
    x1[x*y>0] = np.abs(x1[x*y>0])
    y1[x*y>0] = np.abs(y1[x*y>0])
    x1[x*y<0] = -np.abs(x1[x*y<0])
    y1[x*y<0] = np.abs(y1[x*y<0])
    g = x1+y1*1j
    direction = sob2/sob1
    theta = np.arctan(direction)%(np.pi)
    direction_theta = theta*180/np.pi
    value = {}
    for i in range(36):
        value[i] = []

    for t,v,rr in zip(direction_theta.ravel(), g, r.ravel()):
        value[int(np.around(t/5))%36].append(v*rr)
    sum_value = []
    for key in sorted(value.keys()):
        sum_value.append(np.sum(value[key]))
    sum_value = np.array(sum_value)
    x = sum_value.real
    y = sum_value.imag
    main_d = np.argmax(x**2+y**2)
    sob1 = np.ravel(sob1)
    sob2 = np.ravel(sob1)

    main_x, main_y = sob1[main_d], sob2[main_d]
    return x, y, main_x, main_y


if __name__ == '__main__':
    sigma_vv, sigma_vh, w_speed, w_direction, incidence = get_data(8)
#    plt.imshow(sigma_vv[0])
#    print(w_speed[0])
#    print(w_direction[0])
#    plt.show()
    #img = plt.imread('./sierra.jpg')
    #sigma_vv, sigma_vh, incidence = open_text('subset_0_of_S1B_IW_GRDH_1SDV_20170507T233247_20170507T233312_005499_009A2C_4206_Noise-Cor_Cal_Spk.csv.txt')
    #sigma_vv = np.reshape(sigma_vv, (1241, 1479))
    sigma_vv = sigma_vv[3]

    print(w_direction[3])
    sigma_vv = smoothing.smooth(sigma_vv)
    #sigma_vv = 10 * np.log10(sigma_vv)
    plt.figure(figsize=(20,15))
    plt.subplot(2, 3, 1)
    plt.imshow(sigma_vv, 'Greys')
    plt.title('origin')
    plt.subplot(2, 3, 2)
    sob1, sob2 = sobel(sigma_vv)
    sob1 = smoothing.smooth(sob1)
    sob2 = smoothing.smooth(sob2)
    norm = np.sqrt(sob1**2+sob2**2)
    plt.imshow(sob1, 'Greys')
    plt.title('sobel x')
    plt.subplot(2, 3, 3)
    plt.imshow(sob2, 'Greys')
    plt.title('sobel y')
    plt.subplot(2, 3, 4)
    theta = np.arctan(sob2/sob1)%(np.pi)
    theta = theta*180/np.pi
    plt.hist(theta, bins=36, range=(0, 180))
    plt.title('direction statistic')
    #plt.plot(sob1[2:-2,2:-2], sob2[2:-2],'o')
    plt.subplot(2, 3, 5)
    x, y, main_x, main_y = main_direction(sob1, sob2)
    print(np.arctan(main_y/main_x)%np.pi*180/np.pi)
    plt.plot(x, y)
    plt.subplot(2, 3, 6)
    plt.imshow(sigma_vv, 'Greys')
    plt.quiver(200, 200, main_x, main_y)

    plt.savefig('main_direction.png')
    plt.show()


