

import numpy as np
from scipy import special

class gaussDistribution(object):
    def __init__(self, mu, sigma, skew=0, omega=1):
        self.mu = mu
        self.sigma = sigma
        self.skew = skew
        self.omega = omega

    def pdf(self, data):
        # print(self.skew)
        if self.skew == 0:
            return (1/(self.sigma *np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - self.mu)/self.sigma)**2 )
        else:
            print('test')
            return 2*( (1/(np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data)**2))) * self.cdf(data)

    def cdf(self, data):
        return 0.5*(1 + special.erf(data / np.sqrt(2))) - 2*special.owens_t(data, self.skew)
