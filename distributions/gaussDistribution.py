

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

    def norm_pdf(self, data, mu=None, sigma=None):
        if self.dim == 1:
            return (1/(np.sqrt(2 * np.pi)) ) * np.exp( -0.5 * ( ((data))**2 ) )
        else:
            if mu is None:
                Mu = self.loc
            else:
                Mu = mu
            if sigma is None:
                Sigma = self.Sigma
            else:
                Sigma = sigma 
            if len(data.shape) > 2:
                # print('yes')
                # prob = np.array([]).reshape(data.shape[0], data.shape[1])
                prob = np.empty((data.shape[0], data.shape[1]))
                for i in range(data.shape[0]):
                    cent_data = data[i] - Mu
                    t_prob = (np.matmul(cent_data, np.linalg.inv(Sigma)) * cent_data).sum(axis=1)
                    prob[i,:] = np.exp(-.5 * t_prob) / np.sqrt( ((2*np.pi)**self.dim) * np.abs(np.linalg.det(Sigma)))
            else:
                cent_data = data - Mu
                t_prob = (np.matmul(cent_data, np.linalg.inv(Sigma)) * cent_data).sum(axis=1)
                # print('t_prob: ', t_prob.shape)
                prob = np.exp(-.5 * t_prob) / np.sqrt( ((2*np.pi)**self.dim) * np.abs(np.linalg.det(Sigma)))
                # print(np.sum(prob))
            return prob
            

    def norm_cdf(self, data):
        return 0.5*(1 + special.erf( ( data) / (np.sqrt(2) ) ) )

    def pdf(self, data):
        if isinstance(self.skew, np.ndarray):
            # print('skew: ', self.skew.shape)
            sOmega = self.Sigma - self.skew.T * self.skew
            dDelta = np.identity(self.dim) - self.skew.T * np.linalg.inv(self.Sigma) * self.skew
            tt1 = np.matmul(np.matmul( (data - self.loc), self.skew.T * np.linalg.inv(self.Sigma)), dDelta)
            # tt1 = self.skew * (data - self.loc )/self.omega.T
            print('tt1: ', tt1.shape)
            # prob = (2**self.dim /np.abs(np.linalg.det(self.Sigma))) * self.norm_pdf(data)
            prob = (2**self.dim) * self.norm_pdf(data, sigma=sOmega)
            if len(prob.shape) > 1:
                for i in range(prob.shape[0]):
                    # prob[i,:] = prob[i,:] * self.norm_cdf(tt1[i,:]).prod(axis=1)
                    prob[i,:] = prob[i,:] * self.norm_cdf(tt1[i,:]).prod(axis=1)
            else:
                prob = prob * self.norm_cdf(tt1).prod(axis=1)
            return prob
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
        # print(vals, vecs)
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
        # print(type(rank))
        # print(type(log2pi))
        tre = -0.5 *(rank * log2pi + maha + logdet)
        # print('tre: ', tre)
        return -0.5 * (rank * log2pi + maha + logdet)

