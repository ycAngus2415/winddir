import numpy as np
import cmod5n
'''
polarization correlation coefficient gmf
'''
def PGMF_forward(u, phi,theta):
    rho_0 = 0.05234187*u-0.004635087*u**2+0.00015992*u**3-2.4191*1e-6*u**4+1.3440*1e-8*u**5
    rho = 0.5*rho_0*(1+theta/30)
    ev=np.power(10,-1.5)
    epsion = 1
    delta = 0.5
    cmo5 = cmod5n.cmod5n_forward(u, np.ones(len(u))*60, theta)
    a= np.sqrt(ev)*rho*cmo5/((1+delta)*np.sin(120/180*np.pi)-delta*np.sin(60/180*np.pi))
    alpha1 = -epsion*delta*a
    alpha2 = epsion*(1+delta)*a
    a1=0.58
    a2=0.35
    return alpha1*np.sin(phi/180*np.pi)+alpha2*np.sin(2*phi/180*np.pi)

