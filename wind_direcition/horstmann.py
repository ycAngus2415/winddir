import numpy as np
import matplotlib.pyplot as plt
import scipy


def inverse_horstmann(x):
    return -51.5464 * np.sqrt(-(0.0388 * x)- 0.7773) + 40.4330


def horstmann(x):
    return -0.0097 * x * x+0.7844 * x-35.8912

if __name__ == '__main__':
    mean_vh = np.load('./snap/mean_vh.npy')
    mean_vv = np.load('./snap/mean_vv.npy')
    wsped = np.load('./snap/ten_wind.npy')
    incidence = np.load('./snap/incident.npy')
    mean_vh = mean_vh[mean_vv!=0]
    wsped = wsped[mean_vv!=0]
    incidence = incidence[mean_vv!=0]
    wsped_ = inverse_horstmann(mean_vh)
    rms = np.sqrt(np.sum((wsped - wsped_)*(wsped - wsped_))/len(wsped_))
    print('rms : %f' % rms)
    cov = np.cov(wsped, wsped_)
    var = np.var(wsped)
    var_ = np.var(wsped_)
    relation = cov/np.sqrt(var*var_)
    print(relation)
    print(np.corrcoef(wsped, wsped_))
    print(np.sum(wsped_-wsped)/len(wsped))
    #plt.figure(figsize=(30,19))
    #plt.subplot(1, 2, 1)
    plt.scatter(wsped, wsped_, alpha=0.9, c = np.around(incidence))
    plt.colorbar()
    plt.xlabel('U${_{10}}{^b}$(m/s)')
    plt.ylabel('U${_{10}}{^h}$(m/s)')
    plt.text(11, 28, 'rms = %f'%rms, fontsize=15)
    plt.text(11, 25, 'bias = %f'%(np.sum(wsped_-wsped)/len(wsped)), fontsize=15)
    plt.text(11, 26.5, 'r = %f'%relation[0,1], fontsize=15)
    plt.text(11, 23.5, 'N = %d'%len(wsped_), fontsize=15)
    plt.text(20, 30, 'incidence(°)')
    plt.tick_params(top='off', right='off')
    plt.ylim(0, 29.5)
    #plt.plot(wsped, wsped)
    plt.plot(range(20), range(20), c='#FCC796')
    #plt.subplot(1, 2, 2)
    #x = np.arange(20)
    #y = horstmann(x)
    #plt.scatter(wsped, mean_vh, label='speed from buoy')
    #plt.title('wind and sigma0 relation horstmann')
    #plt.plot(wsped_, mean_vh,'og', label='speed from horstmann')
    #plt.legend()
    #plt.xlabel('wind speed')
    #plt.ylabel('sigma0_vh')
    plt.tick_params(top='off', right='off')
    plt.savefig('correlation_horstmann1.eps')
