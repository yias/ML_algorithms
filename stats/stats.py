import scipy as sp
from scipy import stats as sts
import numpy as np

def covar(data):
    """
    Function for computing the covarianc matrix of the data
    Input: 
        data:       a matrix of the data (number of samples x number of features. 
                    The data should be organized in a column-wise fashion,
                    meaning that each column should corespond to a feature (or
                    dimension) and each row should correspond to a sample (or
                    observation)
    
    Output:
        covMat:     the covariance matrix of the data. an n x n matrix, where n is 
                    the number of features
    """
    mu = np.sum(data, axis=0) / data.shape[0]

    covMatrix = np.dot((data - mu).T , (data - mu)) / (data.shape[0] - 1)

    return covMatrix


def average(data):
    """
    Function for computing the average value (mean) of the data
    Input: 
        data:       a matrix of the data (number of samples x number of features. 
                    The data should be organized in a column-wise fashion,
                    meaning that each column should corespond to a feature (or
                    dimension) and each row should correspond to a sample (or
                    observation)
    
    Output:
        mu:         the average (mean) of the data. an 1 x n vector, where n is 
                    the number of features
    """

    return np.sum(data, axis=0) / data.shape[0]


def std(data):
    """
    Function for computing the standard deviation of the data
    Input: 
        data:       a matrix of the data (number of samples x number of features. 
                    The data should be organized in a column-wise fashion,
                    meaning that each column should corespond to a feature (or
                    dimension) and each row should correspond to a sample (or
                    observation)
    
    Output:
        dSTD:       the standard deviation of the data. an 1 x n vector, where n is 
                    the number of features
    """

    mu = average(data)
    d_m = data - mu
    return np.sqrt(np.diag(np.dot( d_m.T,  d_m) ) / (data.shape[0] - 1) )
