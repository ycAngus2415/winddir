import numpy as np
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt

x = np.arange(-20.1, 20, 0.1)
y = np.arange(-20.1, 20, 0.1)
X,Y = np.meshgrid(x, y)
d = np.dstack((X,Y))
mn = stats.multivariate_normal([0,0],[[225, 0],[0,225]])
#plt.contourf(X,Y, mn.pdf(d))
#plt.colorbar()
#plt.show()
guasss = mn.pdf(d)
cov = [1,-1]
cov = np.reshape(cov,(1, 2))
hx = signal.convolve(guasss, cov, mode='valid')
hx = 10*hx
#hx = -guasss*X/225
fft_hx = np.fft.fft2(hx)
ffts_hx = np.fft.fftshift(fft_hx)
hy = -guasss*Y/225
fft_hy = np.fft.fft2(hy)
ffts_hy = np.fft.fftshift(fft_hy)

img1 = np.log(np.abs(ffts_hx))
img2 = np.log(np.abs(ffts_hy))
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.colorbar()
plt.show()
