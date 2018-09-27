import numpy as np
# Import tensorgen functions and other
import numpy as np
import numexpr as ne

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


def generate_sample(N, M, omega, k):
    """
    Generates a sample of N synthetic samples dustributed as a NBM with binary outcomes
    Returns the generated samples X, with N rows an n columns
    @param N: The number of synthetic samples
    @param M: the centers of the mixture
    @param omega: the mixing weights
    @param k: the number of components
    """
    # Prevent numerical errors
    M[M>0.999999] = 0.999999

    #n is the dimension of the mixture
    n = M.shape[0]

    #x are the cluster assignments
    x = np.random.multinomial(1, omega, N)
    x = np.argmax(x, 1)

    #X is the synthetic dataset
    X = np.zeros((N, n))

    #According to the selected distribution we generate the data
    #M = M / (1 + M.max())
    for i in range(k):
        X[x == i, :] = np.random.binomial(1, M[:, i], [int(sum(x == i)), n])

    return X.astype(float), x

def NaiveBayesClustering(X,k,Eps = 0.01):
    """
    Performs clustering of the dataset
    @param X: the dataset
    @param k: the number of clusters
    @param Eps: The stopping criterion for EM
    """

    #First calculate the parameters of the model
    M,omega = ASVTD(X, k)

    #Use the plugs the parameters into EM
    M, omega, assignment = EM(X,M,omega, Eps)
    #From EM obtains the clustering
    CL = np.argmax(assignment,1)

    return M,omega,CL

def ASVTD(X, k):
    """
    Learn an approximate pair M, omega
    @param X: the dataset
    @param k: the number of clusters
    """
    N, n = X.shape
    if k==1:
        return X.mean(0).reshape(n, 1), np.ones(k)
    E = np.sum(X, 0) / N
    u,s,v = np.linalg.svd(np.transpose(X).dot(X) / N)
    u = u[:,:k].dot((np.diag(np.sqrt(s[:k]))))
    pu = np.linalg.pinv(u)
    Z = pu.dot(X.T)

    HMin = 0
    H = []
    M = np.zeros([n, k])

    for i in range(0, n):
        Y = X[:, i].reshape((N, 1))
        H.append((Z*Y.T).dot(Z.T)/N)

        h, s, v = np.linalg.svd(H[i])
        if np.min(-np.diff(s)) > HMin:
            HMin = np.min(-np.diff(s))
            O = h

    for i in range(0, n):
        s = np.diag(np.transpose(O).dot(H[i]).dot(O))
        M[i, :] = s

    x = np.linalg.lstsq(M, E)
    omega = x[0] ** 2
    omega = omega / sum(omega)

    return M, omega

def numexpr_app(X, a, b):
    XT = X.T
    return ne.evaluate('log(XT * b + a)').sum(0)

def EM(X, M, omega, Eps=0.001, verbose = False):
    """
    Implementation of EM to learn a NBM with binary variables
    @param X: the dataset
    @param M: the centers of the mixture
    @param omega: the mixing weights
    @param Eps: the stopping criterion
    @param verbose: wether to show or not the error
    """

    n,k = M.shape
    N,n = X.shape
    it = 1
    wM = M.copy()
    womega = omega.copy()
    womega[womega<0] = 0.000001
    womega = womega/womega.sum()
    omega_old = womega.copy()+1

    while np.sum(np.abs(womega-omega_old))>Eps:
        assignments = np.zeros((N,k))
        for i in range(k):
            mu = wM[:,i].reshape(n,1)
            mu[mu<=0.00000001] = 0.00000001
            mu[mu>1] = 0.99999

            a = 1 - mu
            b = (2 * mu - 1)

            assignments[:, i] = numexpr_app(X, a, b)+ np.log(womega[i])

        assignments -= np.max(assignments, 1).reshape(len(assignments), 1)
        assignments = np.exp(assignments)
        assignments /= np.sum(assignments,1).reshape(N,1)
        omega_old = womega.copy()
        womega = np.sum(assignments,0)/np.sum(assignments)
        if verbose:
            print(np.sum(np.abs(womega-omega_old)))
            print(womega)
        wM = X.T.dot(assignments)/np.sum(assignments,0)

        it+=1
    return wM,womega,assignments

def generate_sample(N, M, omega, k):
    """
    Generates a sample of N synthetic samples dustributed as a NBM with binary outcomes
    Returns the generated samples X, with N rows an n columns
    @param N: The number of synthetic samples
    @param M: the centers of the mixture
    @param omega: the mixing weights
    @param k: the number of components
    """
    # Prevent numerical errors
    M[M>0.999999] = 0.999999

    #n is the dimension of the mixture
    n = M.shape[0]

    #x are the cluster assignments
    x = np.random.multinomial(1, omega, N)
    x = np.argmax(x, 1)

    #X is the synthetic dataset
    X = np.zeros((N, n))

    #According to the selected distribution we generate the data
    #M = M / (1 + M.max())
    for i in range(k):
        X[x == i, :] = np.random.binomial(1, M[:, i], [int(sum(x == i)), n])

    return X.astype(float), x
