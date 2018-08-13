import numpy as np
import pickle
import matplotlib.pyplot as plt
with open('../data/dataset_ascending/speckle_cal_41043_subset_46.0_ 4.3_4_S1A_IW_GRDH_1SDV_20170526T222855_20170526T222920_016759_01BD7F_85C4.txt', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
vv = data['Sigma0_VV_db:float']
vh = data['Sigma0_VH_db:float']
vv = np.reshape(vv, (400,400))
vh = np.reshape(vh, (400, 400))
vv_vh = vv-vh

#plt.imshow(vv, 'Greys')
vv_vh_fft = np.fft.fft2(vv_vh)
vv_vh_abs = np.abs(vv_vh_fft)
plt.imshow(vv_vh_abs)
plt.show()
