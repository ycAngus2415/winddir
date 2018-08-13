import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#x = np.linspace(0, 1, 1000)
#x = x[:500]
#y = np.sin(4*np.pi*x)+ np.sin(300*np.pi*x)
#y_f = np.fft.fft(y)
#print(np.argmax(np.abs(y_f)))
#print(np.abs(y_f)[1])
#plt.subplot(1, 2, 1)
#plt.plot(x, y)
#plt.subplot(1, 2, 2)
#plt.plot( np.abs(y_f)[:251])
#plt.show()
def sobel(img):
    r = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])/32
    d = r+r.T*1j
    sob1 = signal.convolve2d(img, r,mode='valid')
    sob2 = signal.convolve2d(img, r.T,mode='valid')
    return sob1, sob2

x = [[0, 0,0, 0, 0],[0, 0,1, 0,1], [0, 1, 0,1, 0],[1,0, 1,0,0], [0,0, 0, 0, 0]]
plt.subplot(2, 2, 1)
plt.imshow(x, 'Greys')
x = np.array(x)
#y = np.fft.fft2(x)
#yc = np.fft.fftshift(y)
sob1, sob2 = sobel(x)
plt.subplot(2, 2, 2)
plt.imshow(sob1, 'Greys')
print(sob1)
plt.subplot(2, 2, 3)
plt.imshow(sob2, 'Greys')
print(sob2)
plt.subplot(2,2,4)
sob2=sob2[sob1!=0]
sob1 = sob1[sob1!=0]
theta = np.arctan(sob2/sob1)%np.pi
plt.hist(theta,bins=72,range=(0, np.pi))
print(theta)
plt.show()
