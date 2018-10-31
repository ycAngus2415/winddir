import numpy as np
import matplotlib.pyplot as plt
import scipy


def c_3po(v1, phi):
    print(v1)
    x1 = (0.2983 * v1 - 29.4708)
    x2 = (1+0.07*(phi-34.5)/34.5)
    return np.multiply(x1,x2)

def c_3po_inverse(vh, phi):
    return (vh/(1+0.07*(phi-34.5)/34.5)+29.4708)/0.2983

if __name__ == '__main__':
    mean_vh = np.load('./snap/mean_vh.npy')
    wsped = np.load('./snap/ten_wind.npy')
    incidence = np.load('./snap/incident.npy')
    wsped_ =c_3po_inverse(mean_vh, incidence)
    rms = np.sqrt(np.sum((wsped - wsped_)*(wsped - wsped_))/len(wsped_))
    print('rms : %f' % rms)
    cov = np.cov(wsped, wsped_)
    var = np.var(wsped)
    var_ = np.var(wsped_)
    relation = cov/np.sqrt(var*var_)
    print(relation)
    print(np.corrcoef(wsped, wsped_))
    print(np.sum(wsped_-wsped)/len(wsped))
    plt.figure(figsize=(30,19))
    plt.subplot(1, 2, 1)
    plt.scatter(wsped, wsped_, alpha=0.9)
    plt.title('covariance of wind speed :horstman_vh')
    plt.xlabel('wind speed from buoy')
    plt.ylabel('wind speed from  sigma_vh')
    plt.text(15, 25, 'rms = %f'%rms, fontsize=20)
    plt.text(15, 23, 'bias = %f'%(np.sum(wsped_-wsped)/len(wsped)), fontsize=20)
    plt.text(15, 24, 'r = %f'%relation[0,1], fontsize=20)
    plt.tick_params(top='off', right='off')
    plt.plot(range(20), range(20))
    plt.subplot(1, 2, 2)
    plt.scatter(wsped, mean_vh)
    plt.scatter(wsped_, mean_vh, c=np.around(incidence))
    plt.title('wind and sigma0 relation using c-3po')
    plt.xlabel('wind speed')
    plt.ylabel('sigma0_vh')
    plt.colorbar()
    plt.tick_params(top='off', right='off')
    plt.savefig('correlation_c_3po.png')


