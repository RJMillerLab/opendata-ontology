import numpy as np
from scipy import linalg
from numpy import pi
#from scipy import stats

def entropy(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2*pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2*pi)) + .5*np.log(abs(linalg.det(C)))


def kullback_leibler_divergence(pm, pv, qm, qv):
    eps = 0.0001
    # Determinants of diagonal covariances pv, qv
    dpv = linalg.det(pv)
    dqv = linalg.det(qv)
    # Inverse of diagonal covariance qv
    iqv = linalg.inv(qv)
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log((abs(dqv)+eps) / (abs(dpv)+eps))
             + (np.transpose(iqv * pv).sum())
             + ((diff * iqv * diff).sum())
             - len(pm)))


#pm = np.asarray([0.1, 0.30000000000000004, 0.25])
#pv = np.asarray([[0, 0, 0], [0, 0.020000000000000004, -0.030000000000000002], [0, 0, 0.045000000000000005]])
#qm = np.asarray([0.3, 0.30000000000000004, 0.1])
#qv = np.asarray([[0.08, 0.04, 0], [0, 0.020000000000000004, 0], [0, 0, 0]])
