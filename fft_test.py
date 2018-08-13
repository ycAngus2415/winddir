import numpy as np
import matplotlib.pyplot as plt

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

x = [[0, 0,0, 0, 0],[0, 0,1, 0,1], [0, 1, 0,1, 0],[1,0, 1,0,0], [0,0, 0, 0, 0]]
plt.subplot(2, 2, 1)
plt.imshow(x, 'Greys')
x = np.array(x)
y = np.fft.fft2(x)
yc = np.fft.fftshift(y)
plt.subplot(2, 2, 2)
plt.imshow(np.abs(y), 'Greys')
plt.subplot(2, 2, 3)
plt.imshow(np.log(np.abs(yc)), 'Greys')
plt.colorbar()
plt.show()
