from scipy import signal
from snappy import ProductIO
import snappy
import numpy as np
import cv2
from netCDF4 import Dataset

GeoPos = snappy.jpy.get_type('org.esa.snap.core.datamodel.GeoPos')
PixelPos = snappy.jpy.get_type('org.esa.snap.core.datamodel.PixelPos')
'''
R operation
'''
def R(im):
    by_4 = cv2.getGaussianKernel(7,0)
    bx_4 = np.squeeze(by_4)
    by_2 = cv2.getGaussianKernel(3,0)
    bx_2 = np.squeeze(by_2)
    b_4 = by_4*bx_4
    b_2 = by_2*bx_2
    s_b_2 = signal.convolve2d(im, b_2,mode='same')
    s_b2_b4 = signal.convolve2d(s_b_2, b_4,mode='same')
    return s_b2_b4

'''
new sobel operatoin
'''
def sobel(img):
    r = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])/32
    d = r+r.T*1j
    ss = signal.convolve2d(img, d, mode='same')
    return ss

'''
return corelation param and power R
'''
def measure(sob):
    r_1 = R(np.power(sob,2))
    r_2 = R(np.power(np.abs(sob),2))
    return np.abs(r_1)/r_2, r_1

'''
get ambiguity direction of vv image
'''
def main_lg(sigma_vv, downsize=(200,200),mode='sobel'):
    if mode=='sobel':
        down = cv2.pyrDown(sigma_vv, dstsize=(200,200))
        gg = sobel(down)
    elif mode == 'igl':
        gg = IGL(sigma_vv,7)
    cor, r = measure(gg)
    cor = cor[1:-1,1:-1]
    r = r[1:-1,1:-1]
    cor_dir = r[(cor>=0)*(cor<=1)]
    q = cor_dir.imag/cor_dir.real
    angel= (np.arctan(q)*180/np.pi)%180
    x1, x2 = np.histogram(angel, bins=range(0,181, 5))
    argsort = np.argsort(x1)
    am_direction = x2[argsort[-4:]]
    return am_direction


def f(x,y,s=15):
    return np.exp(-(np.power(x,2)+np.power(y,2))/(2*s**2))*0.5/(np.pi*s**2)
def fft(x0):
    x0x = np.fft.fft2(x0)
    return np.fft.fftshift(x0x)
def ifft(x0):
    x1 = np.fft.ifftshift(x0)
    return np.fft.ifft2(x1)
#def H_x(x,y):
#    return -1j*x*np.exp(-2*np.pi**2*(np.power(x,2)+np.power(y,2)))
#def H_y(x,y):
#    return -1j*y*np.exp(-2*np.pi**2*(np.power(x,2)+np.power(y,2)))
def f_x(x,y,s=15):
    return -x*f(x,y,s)/s**2
def f_y(x,y,s=15):
    return -y*f(x,y,s)/s**2
def L(m,n):
    yn = m+n-1
    return 2**(int(np.log2(yn))+1)

def h_x(l,size, sigma=15):
    x_ = np.linspace(-int(size/2),int(size/2),size)
    y_ = np.reshape(x_, (-1,1))
    filterx = f_x(x_, y_, sigma)
    filterx1 = np.zeros((l,l))
    filterx1[:size,:size] = filterx
    filtery = f_y(x_, y_, sigma)
    filtery1 = np.zeros((l,l))
    filtery1[:size,:size] = filtery
    filter_x = fft(filterx1)
    filter_y = fft(filtery1)
    return filter_x, filter_y

def IGL(image, filter_size=7,sigma=15):
    s = image.shape[0]
    l = L(s, filter_size)
    image1 = np.zeros((l,l))
    image1[:s,:s]=image
    H_x, H_y = h_x(l, filter_size, sigma)
    image_f = fft(image1)
    image_x = image_f * H_x
    image_y = image_f * H_y
    gx = -ifft(image_x).real
    gy = -ifft(image_y).real
    return gx[1:s+1,1:s+1]+1j*gy[1:s+1,1:s+1]

def FFT2(img):
    shape = img.shape
    imgf = np.fft.fft2(img)
    imgfs = np.fft.fftshift(imgf)
    imgfs[int(shape[0]/2-5):int(shape[0]/2+5), int(shape[1]/2-5):int(shape[1]/2+5)]=0
    img_abs = np.abs(imgfs)
    index = np.argmax(img_abs)
    y = int(index/shape[0])
    x = int(index%shape[1])
    dx = x-shape[0]/2
    dy = y-shape[1]/2
    if dx == 0:
        return 90
    else:
        return np.arctan(dy/dx)*180/np.pi

