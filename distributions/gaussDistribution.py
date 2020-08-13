

import numpy as np
from scipy import special
# from scipy.stats import norm
import matplotlib.pyplot as plt
from stats import stats as st


class generalNormDist(object):
    def __init__(self, loc=0, skew=0, omega=1):
        
        if isinstance(loc, list):
            self.loc = np.array(loc)
            self.dim = len(self.loc)
        else:
            self.loc = loc
            if isinstance(self.loc, np.ndarray):
                self.dim = self.loc.size
            else:
                self.dim = 1

        if isinstance(skew, list):
            self.skew = np.array(skew)
        else:
            self.skew = skew

        if self.dim == 1:
            if isinstance(omega, np.ndarray):
                self.omega = omega[0]
            else:
                self.omega = omega
        else:
            if isinstance(omega, np.ndarray):
                if omega.size == 1:
                    self.omega = np.ones((self.dim, 1)) * omega
                else:
                    self.omega = omega
            else:
                omega = np.array(omega)
                if omega.size == 1:
                    self.omega = np.ones((self.dim, 1)) * omega
                else:
                    self.omega = omega
            
            self.Sigma = np.zeros((self.dim, self.dim))
            np.fill_diagonal(self.Sigma, self.omega)

    def norm_pdf(self, data):
        if self.dim == 1:
            return (1/(np.sqrt(2 * np.pi)) ) * np.exp( -0.5 * ( ((data))**2 ) )
        else:
            # n_sigma = st.covar(data)
            # print(n_sigma)
            # n_mu = st.average(data)
            tt = self.logpdf(data)
            print(tt.shape)
            return np.exp(self.logpdf(data))
            # print(data.shape)
            # tt_a = data - self.loc
            # print(tt_a.shape)
            # print(np.linalg.inv(self.Sigma).shape)
            # tt = np.matmul(tt_a, np.linalg.inv(self.Sigma)) * tt_a[:, np.newaxis]
            # print(tt.shape)
            # return (1/np.sqrt( ((2*np.pi)**self.dim) * np.linalg.det(self.Sigma))) * np.exp(-.5*(data - self.loc) * np.linalg.inv(self.Sigma) * (data - self.loc).T)

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
    
    def logpdf(self, x):
        # `eigh` assumes the matrix is Hermitian.
        print(self.Sigma)
        vals, vecs = np.linalg.eigh(self.Sigma)
        # if vals is None:
        # vals = np.ones((self.dim, 1))
        print(vals, vecs)
        logdet     = np.sum(np.log(vals))
        # print('logdet: ', logdet)
        valsinv    = np.array([1./v for v in vals])
        # `vecs` is R times D while `vals` is a R-vector where R is the matrix 
        # rank. The asterisk performs element-wise multiplication.
        U          = vecs * np.sqrt(valsinv)
        # print('U: ', U)
        rank       = len(vals)
        dev        = x - self.loc
        # print('dev: ', dev.shape)
        # print('rank: ', rank)
        # "maha" for "Mahalanobis distance".
        maha       = np.square(np.dot(dev, U)).sum(axis=1)
        # print('maha: ', maha)
        log2pi     = np.log(2 * np.pi)
        # print('log2pi: ', log2pi)
        print(type(rank))
        print(type(log2pi))
        tre = -0.5 *(rank * log2pi + maha + logdet)
        # print('tre: ', tre)
        return -0.5 * (rank * log2pi + maha + logdet)

