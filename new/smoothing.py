import numpy as np
from scipy import signal

def smooth(img):
    B_x = np.array([1, 1])/2
    B_x2 = np.array([1, 2, 1])/4
    B_y2 = np.array([[1],[2],[1]])/4
    B_x_y2 = np.array([[0,0,1],[0,2,0],[1,0,0]])/4
    B_x__y2 = np.array([[1,0,0],[0,2,0],[0,0,1]])/4
    B2 = B_y2*B_x2
    x = np.array([1, 4, 6, 4, 1])/16
    y = np.reshape(x, (5, 1))
    B4 = y*x
    conv1 = signal.convolve2d(img, B2, mode='valid')
    conv2 = signal.convolve2d(conv1, B4, mode='valid')
    return conv2


