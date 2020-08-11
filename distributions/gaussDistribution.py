

import numpy as np
from scipy import special
from scipy.stats import norm
import matplotlib.pyplot as plt

class gaussDistribution(object):
    def __init__(self, mu=0, skew=0, omega=1):
        self.mu = mu
        # self.sigma = sigma
        self.skew = skew
        self.omega = omega

    def norm_pdf(self, data):
        return (1/(np.sqrt(2 * np.pi)) ) * np.exp( -0.5 * ( ((data))**2 ) )

    def norm_cdf(self, data):
        # if data[0]<0:
        #     data = -data
        tt = np.cumsum(self.norm_pdf(data))
        dx = data[1:] - data[0:-1]
        # dx = np.hstack([1,dx])
        # fig, ax = plt.subplots()
        # ax.plot(dx[1:])
        t_r = tt[1:] * dx
        if t_r[-1] < 0:
            return t_r + 1
        else:
            return t_r
        # if tt[np.argmax(np.abs(tt))] < 0:
        #     return - tt * dx
        # else:
            # return tt[1:] * dx

    def pdf(self, data):
        if self.skew == 0:
            return ( ( 1/(self.omega * np.sqrt(2 * np.pi)) ) * np.exp(-0.5 * (( (data - self.mu)/self.omega )**2)))
        else:
            # return ( ( 2/(self.omega * np.sqrt(2 * np.pi)) ) * np.exp(-0.5 * (( (data - self.mu)/self.omega )**2))) *self.norm_cdf(self.skew*data)
            # print(self.norm_cdf(self.skew*data))
            return ( 2/self.omega) * self.norm_pdf(data)[1:0] * self_norm_cdf(self.skew*data)

    def cdf(self, data):
        return 0.5*(1 + special.erf( ( data - self.mu) / (self.omega * np.sqrt(2) ) ) - 2*special.owens_t( (data -  self.mu)/ (self.omega), self.skew))
