
# coding: utf-8

# In[2]:
#the esperiments for wind direction

import snappy
from snappy import ProductIO
import datetime
import sys
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from netCDF4 import Dataset
from scipy import signal
import cv2
import calibrate
import direction
import cmod5n
from mpl_toolkits.basemap import Basemap
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = Dataset('./S1A_IW_GRDH_1SDV_20170518T141604_20170518T141629_016637_01B9B7_C40C.nc')
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
sigma_vv = data.variables['Sigma0_VV'][:]


# In[5]:


print(np.max(lat))
print(np.min(lat))
print(np.max(lon))
print(np.min(lon))


# In[4]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
map_ = Basemap(projection='stere',lat_0=90,lon_0=-105,            llcrnrlat=35.8095 ,urcrnrlat=37.712,            llcrnrlon=-124.916,urcrnrlon=-121.767, resolution='f', ax=ax)
map_.drawmapboundary(fill_color='white')   # 绘制边界
map_.drawcoastlines()
map_.imshow(im, origin='upper', cmap='gray')
plt.show()


# In[6]:


map_(-122.881, 37.363)


# In[ ]:


from PIL import Image


# In[10]:


sigma_vv = data.variables['Sigma0_VV'][:]


# In[3]:


data = Dataset('./S1A_IW_GRDH_1SDV_20170518T141604_20170518T141629_016637_01B9B7_C40C.nc')
sigma_vv = data.variables['Sigma0_VV'][:]
sub = sigma_vv[8000:15000, 10000:24000]
ndbc_dire = (190-318+90)%360


# In[5]:


dire_ = direction.get_dir(sub, sub_size=800, remove_ambi = True, ndbc_= ndbc_dire)
direction.plot_dir(dire_, sub, sub_size=800, file_name='./newwind_4.eps')
plt.arrow(3100, 3100, 200*np.cos(ndbc_dire/180*np.pi), 200*np.sin(ndbc_dire/180*np.pi), color=(0, 0, 1),width=0.04,head_width=60)


# In[2]:


#median filter based on background filed b select the best result between w1 and w2, which have the same shape
#w1 is the initila wind filed
#w3 is the final wind filed
def Median_filter(w1, w2, b, filter_size):
    shape_x = w1.shape[0]
    shape_y = w1.shape[1]
    w3 = np.copy(w1)
    filter_size_h = int(filter_size/2)
    for i in np.arange(filter_size_h, shape_x-filter_size_h, filter_size_h):
        for j in np.arange(filter_size_h, shape_y-filter_size_h, filter_size_h):
            imag = w1[ i-filter_size_h:i+filter_size_h+1, j-filter_size_h:j+filter_size_h+1]
            imag1 = w2[ i-filter_size_h:i+filter_size_h+1, j-filter_size_h:j+filter_size_h+1]
            m = np.median(imag)
            for k in range(filter_size+1):
                for t in range(filter_size+1):
                    t1 = np.abs(imag1[k, t]-m)
                    print('t1:%f'%t1)
                    t2 = np.abs(imag[k, t]-m)
                    print('t2:%f'%t2)
                    if t1>180 :
                        t1=360-t1
                    if t2>180:
                        t2=360-t2
                    if t1<t2:
                        w3[i-filter_size_h+k, j-filter_size_h+t]= imag1[k, t]
        if (shape_y-filter_size_h)%(filter_size_h+1)==0:
            continue
        imag = w1[i-filter_size_h:i+filter_size_h+1, j+filter_size_h+1:]
        imag1 = w2[ i-filter_size_h:i+filter_size_h+1,j+filter_size_h+1:]
        m = np.median(imag)
        for k in range(filter_size+1):
            for t in range((shape_y-filter_size_h)%(filter_size_h+1)):
                t1 = np.abs(imag1[k, t]-m)
                t2 = np.abs(imag[k, t]-m)
                if t1>180 :
                    t1=360-t1
                if t2>180:
                    t2=360-t2
                if t1<t2:
                    w3[i-filter_size_h+k, j+filter_size_h+1+t]= imag1[k, t]
    if (shape_x-filter_size_h)%(filter_size_h+1)!=0:
        for e in np.arange(filter_size_h, shape_y-filter_size_h, filter_size_h):
            imag = w1[i+filter_size_h+1:, e-filter_size_h:j+filter_size_h+1]
            imag1 = w2[i+filter_size_h+1:, e-filter_size_h:j+filter_size_h+1]
            m = np.median(imag)
            for k in range((shape_x-filter_size_h)%(filter_size_h+1)):
                for t in range(filter_size+1):
                    t1 = np.abs(imag1[k, t]-m)
                    t2 = np.abs(imag[k, t]-m)
                    if t1>180 :
                        t1=360-t1
                    if t2>180:
                        t2=360-t2
                    if t1<t2:
                        w3[i+filter_size_h+k+1, e-filter_size_h+t]= imag1[k, t]
        if (shape_y-filter_size_h)%(filter_size_h+1)!=0:
            imag = w1[i+filter_size_h+1:, j+filter_size_h+1:]
            imag1 = w2[i+filter_size_h+1:, j+filter_size_h+1:]
            m = np.median(imag)
            for k in range((shape_x-filter_size_h)%(filter_size_h+1)):
                    for t in range((shape_y-filter_size_h)%(filter_size_h+1)):
                        t1 = np.abs(imag1[k, t]-m)
                        t2 = np.abs(imag[k, t]-m)
                        if t1>180 :
                            t1=360-t1
                        if t2>180:
                            t2=360-t2
                        if t1<t2:
                            w3[i+filter_size_h+k+1, j+filter_size_h+t+1]= imag1[k, t]
    return w3


# In[8]:


ww3 = Median_filter(w1=ww3, w2=dire_[1], b=ndbc_dire, filter_size=8)


# In[9]:


direction.plot_dir((ww3,), sub[:, 7000:], sub_size=400, file_name='./newwind_4h.eps')


# In[11]:


for i in np.arange(3, 10, 2):
    print(i)


# In[18]:


plt.imshow(-10*np.log10(sigma_vv), 'gray')


# In[24]:


plt.imshow(out)


# In[12]:


10.9


# In[4]:


360-8.83


# In[31]:


im.shape


# In[14]:


plt.imshow(-10*np.log10(im), 'gray')


# In[36]:


im[0, ]


# In[39]:


im[15000,15000]


# In[38]:


sigma_vv[0, 0]


# In[40]:


im[im<0]=0


# In[42]:


np.max(im)


# In[43]:


im[im==0]=100


# In[6]:


map_ = Basemap(projection='stere',lat_0=np.max(lat),lon_0=np.min(lon),            llcrnrlat=np.min(lat) ,urcrnrlat=np.max(lat),            llcrnrlon=np.min(lon),urcrnrlon=np.max(lon), resolution='f')


# In[16]:


sar_ndbc = pd.read_csv('./sar_ndbc.csv')


# In[19]:


sar_ndbc[sar_ndbc['ndbc']==ndbc_name][sar_ndbc['sar_name']==sar_name]


# In[20]:


ndbc_i = pd.read_csv('/Volumes/Yangchao/data/ndbc_station_information.csv')


# In[25]:


line = ndbc_i[ndbc_i['name']=='46012']


# In[27]:


line.lon, line.lat


# In[29]:


np.min(lat)


# In[30]:


np.max(lat)


# In[31]:


np.min(lon)


# In[32]:


np.max(lon)


# In[34]:


0.881*60


# In[35]:


0.86*60


# In[36]:


0.363*60


# In[ ]:


7824 12046


# In[37]:


0.78*60


# In[41]:


fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(-10*np.log10(im), 'gray')


# In[2]:


data = Dataset('./S1A_IW_GRDH_1SDV_20170518T141604_20170518T141629_016637_01B9B7_C40C.nc')
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
sigma_vv = data.variables['Sigma0_VV'][:]


# In[15]:


from PIL import Image
image = Image.fromarray(sigma_vv)
out = image.transpose(Image.FLIP_LEFT_RIGHT)
out = out.rotate(351, expand=1)
im = np.array(out)


# In[16]:


map_ = Basemap(projection='stere',lat_0=np.max(lat),lon_0=np.min(lon),            llcrnrlat=np.min(lat) ,urcrnrlat=np.max(lat),            llcrnrlon=np.min(lon),urcrnrlon=np.max(lon), resolution='f')


# In[69]:


x, y = map_(-122.881,  37.363)
sub_lat = lat[9800:14210,15950:20360]
sub_lon = lon[9800:14210,15950:20360]
poly_lat = [sub_lat[0,0],sub_lat[0, -1],sub_lat[-1,-1],sub_lat[-1, 0], sub_lat[0,0]]
poly_lon =  [sub_lon[0,0],sub_lon[0, -1],sub_lon[-1,-1],sub_lon[-1, 0], sub_lon[0,0]]
xx, yy = map_(poly_lon, poly_lat)


# In[70]:



map_.drawcoastlines()
map_.imshow(-10*np.log10(im),origin='upper',cmap='gray')
parallels=np.arange(np.min(lat), np.max(lat), 0.5)
map_.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(np.min(lon), np.max(lon), 0.8)
map_.drawmeridians(meridians,labels=[0,0,0,1])
map_.drawmapboundary(fill_color='white')
map_.plot(int(x), int(y), marker='D', color='r')
map_.plot(np.around(xx), np.around(yy),'g-')
plt.text(int(x+10), int(y+10), '46012', color='white')
plt.savefig('./1.png', dpi=400)


# In[16]:


im.shape


# In[2]:


data = Dataset('./S1A_IW_GRDH_1SDV_20170518T233538_20170518T233603_016643_01B9E6_D58F.nc')
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
sigma_vv = data.variables['Sigma0_VV'][:]
from PIL import Image
image = Image.fromarray(sigma_vv)
out = image.transpose(Image.FLIP_TOP_BOTTOM)
out = out.rotate(10.5, expand=1)
im = np.array(out)


# In[3]:


map_ = Basemap(projection='stere',lat_0=np.max(lat),lon_0=np.min(lon)+0.004,            llcrnrlat=np.min(lat) ,urcrnrlat=np.max(lat),            llcrnrlon=np.min(lon)+0.004,urcrnrlon=np.max(lon)+0.002, resolution='f')


# In[4]:


x, y = map_(-82.773,24.693)
sub_lat = lat[11400:14100,7150:10450]
sub_lon = lon[11400:14100,7150:10450]
poly_lat = [sub_lat[0,0],sub_lat[0, -1],sub_lat[-1,-1],sub_lat[-1, 0], sub_lat[0,0]]
poly_lon =  [sub_lon[0,0],sub_lon[0, -1],sub_lon[-1,-1],sub_lon[-1, 0], sub_lon[0,0]]
xx, yy = map_(poly_lon, poly_lat)


# In[5]:


map_.drawcoastlines(linewidth=0.1)
map_.imshow(-10*np.log10(im),origin='upper',cmap='gray')
parallels=np.arange(np.min(lat), np.max(lat), 0.5)
map_.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(np.min(lon), np.max(lon), 0.8)
map_.drawmeridians(meridians,labels=[0,0,0,1])
map_.drawmapboundary(fill_color='white', linewidth=0.1)
map_.plot(int(x), int(y), marker='D', color='r')
map_.plot(np.around(xx), np.around(yy),'g-')
plt.text(int(x+10), int(y+10), 'plsf1', color='white')
plt.savefig('./2.png', dpi=400)


# In[23]:


data = Dataset('./S1A_IW_GRDH_1SDV_20170520T001818_20170520T001843_016658_01BA5D_566B.nc')
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
sigma_vv = data.variables['Sigma0_VV'][:]
from PIL import Image
image = Image.fromarray(sigma_vv)
out = image.transpose(Image.FLIP_TOP_BOTTOM)
out = out.rotate(10.5, expand=1)
im = np.array(out)


# In[24]:


map_ = Basemap(projection='stere',lat_0=np.max(lat),lon_0=np.min(lon),            llcrnrlat=np.min(lat) ,urcrnrlat=np.max(lat),            llcrnrlon=np.min(lon),urcrnrlon=np.max(lon), resolution='f')


# In[25]:


x, y = map_(-94.033,29.683)


# In[31]:


map_.drawcoastlines()
map_.imshow(-10*np.log10(im),origin='upper',cmap='gray')
parallels=np.arange(np.min(lat), np.max(lat), 0.5)
map_.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(np.min(lon), np.max(lon), 0.8)
map_.drawmeridians(meridians,labels=[0,0,0,1])
map_.drawmapboundary(fill_color='white')
map_.plot(int(x), int(y), marker='D', color='r')
plt.text(int(x-30), int(y+30), 'srst2', color='white')
plt.savefig('./3.png', dpi=400)


# In[30]:


plt.imshow(-10*np.log10(im), 'gray')


# In[4]:


sub_lat = lat[10200:14000,15800:20500]
sub_lon = lon[10200:14000,15800:20500]
poly_lat = [sub_lat[0,0],sub_lat[0, -1],sub_lat[-1,-1],sub_lat[-1, 0],sub_lat[0,0]]
poly_lon =  [sub_lon[0,0],sub_lon[0, -1],sub_lon[-1,-1],sub_lon[-1, 0],sub_lon[0,0]]


# In[8]:


sub_lat[-1,-1]


# In[9]:


sub_lat[-1, 0]


# In[121]:



sub_lat = lat[9800:14210,15950:20360]
sub_lon = lon[9800:14210,15950:20360]
poly_lat = [sub_lat[0,0],sub_lat[0, -1],sub_lat[-1,-1],sub_lat[-1, 0],sub_lat[0,0]]
poly_lon =  [sub_lon[0,0],sub_lon[0, -1],sub_lon[-1,-1],sub_lon[-1, 0],sub_lon[0,0]]
sub = sigma_vv[9800:14210,15950:20360]

dire_ = direction.get_dir(sub, sub_size=500, remove_ambi = True, ndbc_= ndbc_dire)

sub_size=500
sub_size_h = int(sub_size/2)
shape_x = sub.shape[0]
shape_y = sub.shape[1]
x_, y_ = np.arange(sub_size_h, shape_x-sub_size_h, sub_size), np.arange(sub_size_h, shape_y-sub_size_h, sub_size)
points = np.meshgrid(x_, y_)
X, Y = map_1(sub_lon[points], sub_lat[points])

image1 = Image.fromarray(sub)
out1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
out1 = out1.rotate(351, expand=1)
sub= np.array(out1)



map_1 = Basemap(projection='stere',lat_0=np.max(sub_lat),lon_0=np.min(sub_lon),            llcrnrlat=np.min(sub_lat) ,urcrnrlat=np.max(sub_lat),            llcrnrlon=np.min(sub_lon),urcrnrlon=np.max(sub_lon), resolution='f')


index = 0
color_list = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]







# In[126]:


map_1.drawcoastlines()
map_1.imshow(-10*np.log10(sub),origin='upper',cmap='gray')
parallels=np.arange(np.min(sub_lat), np.max(sub_lat), 0.2)
map_1.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(np.min(sub_lon), np.max(sub_lon), 0.2)
map_1.drawmeridians(meridians,labels=[0,0,0,1])
map_1.drawmapboundary(fill_color='white')
index=0
for d in dire_:
    U1 = 20*np.sin((280-d)%360/180*np.pi)
    V1 = 20*np.cos((280-d)%360/180*np.pi)
    map_1.quiver(X, Y, U1[points], V1[points], width=0.004, color=color_list[index])
    index+=1
map_1.quiver([[3000]], [3000], [[20*np.sin(318/180*np.pi)]], [[20*np.cos(318/180*np.pi)]], width=0.004, color=(0,0, 1))
plt.savefig('./1_wind_5.png', dpi=400)


# In[124]:


ndbc_dire = (190-318+90)%360
ww3 = Median_filter(w1=dire_[0][points], w2=dire_[1][points], b=ndbc_dire, filter_size=4)


# In[127]:


d = ww3
U1 = 20*np.sin((280-d)%360/180*np.pi)
V1 = 20*np.cos((280-d)%360/180*np.pi)

map_1.drawcoastlines()
map_1.imshow(-10*np.log10(sub),origin='upper',cmap='gray')
parallels=np.arange(np.min(sub_lat), np.max(sub_lat), 0.2)
map_1.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(np.min(sub_lon), np.max(sub_lon), 0.2)
map_1.drawmeridians(meridians,labels=[0,0,0,1])
map_1.drawmapboundary(fill_color='white')

map_1.quiver(X, Y, U1, V1, width=0.004, color=(0, 1, 0))
map_1.quiver([[3000]], [3000], [[20*np.sin(318/180*np.pi)]], [[20*np.cos(318/180*np.pi)]], width=0.004, color=(0,0, 1))
plt.savefig('./1_wind_m_5.png', dpi=400)


# plot new d58f

# In[10]:


ndbc_dire = (350-97+90)%360
sub_lat = lat[11400:14100,7150:10450]
sub_lon = lon[11400:14100,7150:10450]
poly_lat = [sub_lat[0,0],sub_lat[0, -1],sub_lat[-1,-1],sub_lat[-1, 0],sub_lat[0,0]]
poly_lon =  [sub_lon[0,0],sub_lon[0, -1],sub_lon[-1,-1],sub_lon[-1, 0],sub_lon[0,0]]
sub = sigma_vv[11400:14100,7150:10450]

dire_ = direction.get_dir(sub, sub_size=100, remove_ambi = True, ndbc_= ndbc_dire)

sub_size=100
sub_size_h = int(sub_size/2)
shape_x = sub.shape[0]
shape_y = sub.shape[1]
x_, y_ = np.arange(sub_size_h, shape_x-sub_size_h, sub_size), np.arange(sub_size_h, shape_y-sub_size_h, sub_size)
points = np.meshgrid(x_, y_)

image1 = Image.fromarray(sub)
out1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
out1 = out1.rotate(10.5, expand=1)
sub= np.array(out1)

map_1 = Basemap(projection='stere',lat_0=np.max(sub_lat),lon_0=np.min(sub_lon),            llcrnrlat=np.min(sub_lat) ,urcrnrlat=np.max(sub_lat),            llcrnrlon=np.min(sub_lon),urcrnrlon=np.max(sub_lon), resolution='f')
X, Y = map_1(sub_lon[points], sub_lat[points])

index = 0
color_list = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]


# In[7]:


map_1.drawcoastlines()
map_1.imshow(-10*np.log10(sub),origin='upper',cmap='gray')
parallels=np.arange(np.min(sub_lat), np.max(sub_lat), 0.1)
map_1.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(np.min(sub_lon), np.max(sub_lon), 0.1)
map_1.drawmeridians(meridians,labels=[0,0,0,1])
map_1.drawmapboundary(fill_color='white')
index=0
for d in dire_:
    U1 = 20*np.sin((80-d)%360/180*np.pi)
    V1 = 20*np.cos((80-d)%360/180*np.pi)
    map_1.quiver(X, Y, U1[points], V1[points], width=0.004, color=color_list[index])
    index+=1
map_1.quiver([[3000]], [3000], [[20*np.sin(97/180*np.pi)]], [[20*np.cos(97/180*np.pi)]], width=0.004, color=(0,0, 1))
plt.savefig('./2_wind_2.png', dpi=400)


# In[19]:


import direction
ww3 = direction.Median_filter(w1=ww3, w2=dire_[1][points], b=ndbc_dire, filter_size=4)
d = ww3


# In[24]:


d=ww3
U1 = 20*np.sin((80-d)%360/180*np.pi)
V1 = 20*np.cos((80-d)%360/180*np.pi)

map_1.drawcoastlines()
map_1.imshow(-10*np.log10(sub),origin='upper',cmap='gray')
parallels=np.arange(np.min(sub_lat), np.max(sub_lat), 0.1)
map_1.drawparallels(parallels,labels=[1,0,0,0])
meridians = np.arange(np.min(sub_lon), np.max(sub_lon), 0.1)
map_1.drawmeridians(meridians,labels=[0,0,0,1])
map_1.drawmapboundary(fill_color='white')


map_1.quiver(X, Y, U1, V1, width=0.004, color=(0, 1, 0))
d=dire_[0]
U1 = 20*np.sin((80-d)%360/180*np.pi)
V1 = 20*np.cos((80-d)%360/180*np.pi)
map_1.quiver(X, Y, U1[points], V1[points], width=0.004, color=(1,1,1))
map_1.quiver([[2000]], [2000], [[20*np.sin(97/180*np.pi)]], [[20*np.cos(97/180*np.pi)]], width=0.004, color=(0,0, 1))
plt.savefig('./2_wind_a_1_3.png', dpi=400)


# In[113]:


data = Dataset('./S1A_IW_GRDH_1SDV_20170518T141604_20170518T141629_016637_01B9B7_C40C.nc')
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
sigma_vv = data.variables['Sigma0_VV'][:]
sub = sigma_vv[9800:14210,15950:20360]
ndbc_dire = (190-318+90)%360
dire_ = direction.get_dir(sub, sub_size=500, remove_ambi = True, ndbc_= ndbc_dire)
d=[]
sub_size=500
sub_size_h = int(sub_size/2)
shape_x = sub.shape[0]
shape_y = sub.shape[1]
x_, y_ = np.arange(sub_size_h, shape_x-sub_size_h, sub_size), np.arange(sub_size_h, shape_y-sub_size_h, sub_size)
points = np.meshgrid(x_, y_)
for d1 in dire_:
    w = (280-d1)%360
    d.append(np.ravel(w[points]))


# In[114]:


ww3 = direction.Median_filter(w1=dire_[0][points], w2=dire_[1][points], b=ndbc_dire, filter_size=4)


# In[51]:


ww3 = direction.Median_filter(w1=dire_[0][points], w2=w2, b=ndbc_dire, filter_size=4)


# In[115]:


w =(280-np.ravel(ww3))%360


# In[50]:


w2 =(280-ww3)%360


# In[116]:


n=np.abs(d[0]-318)
n[n>180]=360-np.abs(d[0][n>180]-318)
print(np.sum(n)/len(d[0]))
print(np.sqrt(np.sum(np.power(n,2))/len(n)))
m= np.arctan2(np.sum(np.sin(d[0]/180*np.pi))/len(d[0]), np.sum(np.cos(d[0]/180*np.pi))/len(d[0]))/np.pi*180
print(m%360)
nn = np.abs(d[0]-m)
if len(nn[nn>180])!=0:
    nn[nn>180]=360-np.abs(d[0][nn>180]-m)
print(np.sqrt(np.sum(np.power(nn,2))/len(nn)))


# In[117]:


n=np.abs(d[1]-318)
n[n>180]=360-np.abs(d[1][n>180]-318)
print(np.sum(n)/len(d[1]))
print(np.sqrt(np.sum(np.power(n,2))/len(n)))
m= np.arctan2(np.sum(np.sin(d[1]/180*np.pi))/len(d[1]), np.sum(np.cos(d[1]/180*np.pi))/len(d[1]))/np.pi*180
print(m%360)
nn = np.abs(d[1]-m)
if len(nn[nn>180])!=0:
    nn[nn>180]=360-np.abs(d[1][nn>180]-m)
print(np.sqrt(np.sum(np.power(nn,2))/len(nn)))


# In[118]:


n=np.abs(w-318)
n[n>180]=360-np.abs(w[n>180]-318)
print(np.sum(n)/len(w))
print(np.sqrt(np.sum(np.power(n,2))/len(n)))
m= np.arctan2(np.sum(np.sin(w/180*np.pi))/len(w), np.sum(np.cos(w/180*np.pi))/len(w))/np.pi*180
print(m%360)
nn = np.abs(w-m)
if len(nn[nn>180])!=0:
    nn[nn>180]=360-np.abs(w[nn>180]-m)
print(np.sqrt(np.sum(np.power(nn,2))/len(nn)))


# In[119]:



plt.plot(d[0], np.ones(len(d[0])),'s',markersize=2)
plt.plot(d[1], np.ones(len(d[1]))*1.5,'o',markersize=2)
plt.plot(w, np.ones(len(w))*2,'*',markersize=2)
plt.ylim(0.5,2.5)
plt.yticks((1,1.5,2),('FFT','LG','Combine'))
plt.plot(318,1,'ro',label='NDBC wind direction')
plt.plot(318,1.5,'ro')
plt.plot(318,2,'ro')
plt.legend()
plt.xlabel('wind direction(°)')
plt.savefig('1_result.eps')


# 第二幅图

# In[107]:


data = Dataset('./S1A_IW_GRDH_1SDV_20170518T233538_20170518T233603_016643_01B9E6_D58F.nc')
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
sigma_vv = data.variables['Sigma0_VV'][:]
sub = sigma_vv[11400:14100,7150:10450]
ndbc_dire = (350-97+90)%360
dire_ = direction.get_dir(sub, sub_size=500, remove_ambi = True, ndbc_= ndbc_dire)
sub_size=500
sub_size_h = int(sub_size/2)
shape_x = sub.shape[0]
shape_y = sub.shape[1]
x_, y_ = np.arange(sub_size_h, shape_x-sub_size_h, sub_size), np.arange(sub_size_h, shape_y-sub_size_h, sub_size)
points = np.meshgrid(x_, y_)
d2=[]
for d1 in dire_:
    w=(80-d1)%360
    d2.append(np.ravel(w[points]))


# In[108]:


ww1 = direction.Median_filter(w1=dire_[0][points], w2=dire_[1][points], b=ndbc_dire, filter_size=4)


# In[ ]:


ww1 = direction.Median_filter(w1=dire_[0][points], w2=w3, b=ndbc_dire, filter_size=4)


# In[109]:


w1 =(80-np.ravel(ww1))%360


# In[ ]:


w3 =(80-ww1)%360


# In[110]:


n=np.abs(d2[0]-97)
n[n>180]=360-np.abs(d2[0][n>180]-97)
me = np.sum(n)/len(d2[0])
print(me)
print(np.sqrt(np.sum(np.power(n,2))/len(n)))
m= np.sum(np.arctan2(np.sin(d2[0]/180*np.pi), np.cos(d2[0]/180*np.pi))/np.pi*180)/len(d2[0])
print(m)
nn = np.abs(d2[0]-m)
if len(nn[nn>180])!=0:
    nn[nn>180]=360-np.abs(d2[0][nn>180]-m)
print(np.sqrt(np.sum(np.power(nn,2))/len(nn)))


# In[111]:


n=np.abs(d2[1]-97)
n[n>180]=360-np.abs(d2[0][n>180]-97)
print(np.sum(n)/len(d2[1]))
print(np.sqrt(np.sum(np.power(n,2))/len(n)))
m= np.sum(np.arctan2(np.sin(d2[1]/180*np.pi), np.cos(d2[1]/180*np.pi))/np.pi*180)/len(d2[1])
print(m)
nn = np.abs(d2[1]-m)
if len(nn[nn>180])!=0:
    nn[nn>180]=360-np.abs(d2[1][nn>180]-m)
print(np.sqrt(np.sum(np.power(nn,2))/len(nn)))


# In[112]:


n=np.abs(w1-97)
n[n>180]=360-np.abs(d2[0][n>180]-97)
print(np.sum(n)/len(w1))
print(np.sqrt(np.sum(np.power(n,2))/len(n)))
m= np.sum(np.arctan2(np.sin(w1/180*np.pi), np.cos(w1/180*np.pi))/np.pi*180)/len(w1)
print(m)
nn = np.abs(w1-m)
if len(nn[nn>180])!=0:
    nn[nn>180]=360-np.abs(w1[nn>180]-m)
print(np.sqrt(np.sum(np.power(nn,2))/len(nn)))


# In[91]:


plt.plot(d2[0], np.ones(len(d2[0])),'*',markersize=4)
plt.plot(d2[1], np.ones(len(d2[1]))*1.5,'s',markersize=4)
plt.plot(w1, np.ones(len(w1))*2,'o',markersize=4)
plt.ylim(0.5,2.5)
plt.yticks((1,1.5,2),('FFT','LG','Combine'))
plt.plot(97,1,'ro',label='NDBC wind direction')
plt.plot(97,1.5,'ro')
plt.plot(97,2,'ro')
plt.legend()
plt.xlabel('wind direction(°)')
plt.savefig('2_result.eps')


# In[36]:





# In[37]:





# In[46]:


print(d2[1])


# In[74]:


np.arctan2(np.sin(10/180*np.pi),np.cos(10/180*np.pi))/np.pi*180

