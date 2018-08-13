import numpy as np
import matplotlib.pyplot as plt

# wind speed function ,
# x is wind speed
# theta is incidence angle
# phi is wind direction
def f_wind(x, theta, phi):
    if (theta <=36 and theta >30) and (x>8 and x<=12.3):
        return 0.46*x-34.06
    elif (theta <=36 and theta >30) and (x>12.3):
        return 0.89*x-39.36
    elif (theta >36 and theta < 41) and (x>9.2):
        return 0.73*x-38.08


def f_incidence(x, theta, phi):
     if (theta <=36 and theta >30) and (x>8 and x<=12.3):
        return 0.13*theta*theta-0.82*theta+103.88
     elif (theta <=36 and theta >30) and (x>12.3):
        return 0.08*theta*theta-4.86*theta+48.97
     elif (theta >36 and theta < 41) and (x>9.2):
         return 0.16*theta*theta-12.10*theta+195.98

def w(theta):
    if  (theta <=36 and theta >30) :
        return -0.039
    elif (theta >36 and theta < 41) :
        return -0.045

def c(theta):
    if  (theta <=36 and theta >30) :
        return -0.32
    elif (theta >36 and theta < 41) :
        return -0.68

def a(phi):
    if (phi>45 and phi< 135) or (phi>225 and phi < 315):
        return -0.5
    else:
        return 0.5

def f(x, theta, phi):
    wind_function = []
    incidence_function = []
    w_function = []
    c_function = []
    a_function = []
    for i in range(len(x)):
        wind_function.append(f_wind(x[i], theta[i], phi[i]))
        incidence_function.append(f_incidence(x[i], theta[i], phi[i]))
        w_function.append(w(theta[i]))
        c_function.append(c(theta[i]))
        a_function.append(a(phi[i]))
    wind_function = np.array(wind_function)
    incidence_function = np.array(incidence_function)
    w_function = np.array(w_function)
    c_function = np.array(c_function)
    a_function = np.array(a_function)
    return wind_function * (1 + w_function*(2*incidence_function/(np.max(incidence_function)-np.min(incidence_function))-1))+c_function+a_function

def f_inverse(sigma, theta, phi):
    v1 = 10*np.ones(len(sigma))
    step = 0.1
    for i in range(200):
        sigma_ = f(v1, theta, phi)
        ind = sigma_<sigma
        v1[ind]= v1[ind]+step
        ind = sigma_>sigma
        v1[ind]= v1[ind]-step
    return v1


if __name__ == '__main__':
    wind_speed = np.load('../data/data/ten_wind.npy')
    wind_incidence = np.load('../data/data/incident.npy')
    wind_direction = np.load('../data/data/wdir_relative.npy')
    sigma = np.load('../data/data/mean_vh.npy')
    wind_incidence = wind_incidence[wind_speed>9.2]
    wind_direction = wind_direction[wind_speed>9.2]
    sigma = sigma[wind_speed>9.2]
    wind_speed = wind_speed[wind_speed>9.2]

    wind_direction = wind_direction[(wind_incidence>30)*(wind_incidence<41)]
    sigma = sigma[(wind_incidence>30)*(wind_incidence<41)]
    wind_speed = wind_speed[(wind_incidence>30)*(wind_incidence<41)]
    wind_incidence = wind_incidence[(wind_incidence>30)*(wind_incidence<41)]
    print(f([10, 15], [35, 39], [45, 180]))
    speed = f_inverse(sigma, wind_incidence, wind_direction)
    rms = np.sqrt(np.sum((wind_speed -speed)**2)/len(wind_speed))
    print(rms)
    print(np.corrcoef(speed, wind_speed))
    plt.subplot(1, 2, 1)
    plt.scatter(wind_speed, sigma, c=wind_incidence)
    plt.subplot(1, 2, 2)
    plt.scatter(wind_speed, speed, c=wind_incidence)
    plt.plot([8,20], [8, 20])
    plt.text(15, 20, 'rms=%f'%rms, fontsize=15)
    plt.text(15, 18.5, 'r=%f'%np.corrcoef(speed, wind_speed)[0,1],fontsize=15)
    plt.text(15, 17, 'N=%d'%len(wind_speed),fontsize=15)
    plt.savefig('Huang.png')
    plt.show()

