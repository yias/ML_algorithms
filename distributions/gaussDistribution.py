

import numpy as np
from scipy import special
from scipy.stats import norm
import matplotlib.pyplot as plt

class generalNormDist(object):
    def __init__(self, loc=0, skew=0, omega=1):
        self.loc = loc
        # self.sigma = sigma
        self.skew = skew
        self.omega = omega

    def norm_pdf(self, data):
        return (1/(np.sqrt(2 * np.pi)) ) * np.exp( -0.5 * ( ((data))**2 ) )

    def norm_cdf(self, data):
        # tt = np.cumsum(self.norm_pdf(data))
        # dx = data[1:] - data[0:-1]
        # dx = np.hstack([0,dx])
        # t_r = tt * dx
        # if t_r[-1] < 0:
        #     return t_r + 1
        # else:
        #     return t_r
        return 0.5*(1 + special.erf( ( data) / (np.sqrt(2) ) ) )

    def pdf(self, data):
        if self.skew == 0:
            # return ( ( 1/(self.omega * np.sqrt(2 * np.pi)) ) * np.exp(-0.5 * (( (data - self.loc)/self.omega )**2)))
            return self.norm_pdf((data - self.loc)/self.omega)
        else:
            return ( 2/self.omega) * self.norm_pdf((data - self.loc)/self.omega) * self.norm_cdf(self.skew*(data - self.loc)/self.omega)

    def cdf(self, data):
        return 0.5*(1 + special.erf( ( data - self.loc) / (self.omega * np.sqrt(2) ) ) - 2*special.owens_t( (data -  self.loc)/ (self.omega), self.skew))

    def loglkd(self, data):
        tt = np.log(self.pdf(data))
        return np.log(self.pdf(data))
        # return np.cumsum(tt)
