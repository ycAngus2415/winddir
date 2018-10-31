import numpy as np
import matplotlib.pyplot as plt
#test vh branch dev
#test another branch dev
sigma_vh = np.load('../data/data/mean_vh.npy')
sigma_vv = np.load('../data/data/mean_vv.npy')
incidence = np.load('../data/data/incident.npy')
wind_speed = np.load('../data/data/ten_wind.npy')
wind_direction = np.load('../data/data/wdir_relative.npy')
sigma_vh = sigma_vh[sigma_vv!=0]
incidence = incidence[sigma_vv!=0]
wind_speed = wind_speed[sigma_vv!=0]
wind_direction = wind_direction[sigma_vv!=0]
sigma_vv = sigma_vv[sigma_vv!=0]
sigma_vh1 = np.power(10, sigma_vh/10)
sigma_vv1 = np.power(10, sigma_vv/10)
plt.subplot(3,2,1)
plt.scatter(wind_speed,sigma_vv, c=incidence)
plt.subplot(3, 2, 2)
plt.scatter(wind_direction, sigma_vv, c=incidence)
plt.subplot(3, 2, 3)
plt.scatter(wind_speed, 10*np.log10(sigma_vh1/sigma_vv1),c=incidence)
plt.subplot(3, 2, 4)
plt.scatter(wind_direction, 10*np.log10(sigma_vh1/sigma_vv1),c=incidence)
plt.subplot(3, 2, 5)
plt.scatter(incidence, 10*np.log10(sigma_vh1/sigma_vv1),c=wind_speed)
plt.show()
