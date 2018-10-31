import numpy as np
import matplotlib.pyplot as plt
import scipy


def inverse_c_2po(x):
    return (x + 35.652)/0.580


def c_2po(x):
    return 0.580*x - 35.652

if __name__ == '__main__':
    wsped_ = inverse_c_2po(mean_vh)
    mean_vh = np.load('./snap/mean_vh.npy')
    wsped = np.load('./snap/ten_wind.npy')
    rms = np.sqrt(np.sum((wsped - wsped_)*(wsped - wsped_))/len(wsped_))
    print('rms : %f'%rms)
    ws = np.concatenate((np.reshape(wsped,(1, len(wsped))), np.reshape(wsped_, (1, len(wsped_)))), axis=0)
    print(ws.shape)
    cov = np.cov(ws)
    var = np.var(wsped)
    var_ = np.var(wsped_)
    relation = cov/np.sqrt(var*var_)
    print(relation)
    print(np.corrcoef(wsped, wsped_))
    print(np.sum(wsped_-wsped)/len(wsped))
    np.cov(ws)
    plt.figure(figsize=(30, 15))
    plt.subplot(1, 2, 1)
    plt.scatter(wsped, wsped_, alpha=0.9)
    plt.title('correlation c_2po')
    plt.xlabel('wind speed from buoy')
    plt.ylabel('wind speed from  sigma_vh')
    plt.text(12, 20, 'rms = %3f'%rms, fontsize=15)
    plt.text(12, 18, 'bias = %f'%(np.sum(wsped_-wsped)/len(wsped)), fontsize=15)
    plt.text(12, 19, 'r = %3f'%relation[0,1], fontsize=15)
    plt.tick_params(top='off', right='off')
    plt.plot(wsped, wsped)
    plt.subplot(1, 2, 2)
    x = np.arange(20)
    y = c_2po(x)
    plt.scatter(wsped, mean_vh)
    plt.plot(x, y)
    plt.title('wind and sigma0 relation c_2po')
    plt.xlabel('wind speed')
    plt.ylabel('sigma0_vh')
    plt.tick_params(top='off', right='off')
    plt.savefig('correlation_c_2po.png')


