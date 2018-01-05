import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import peakutils

def eigenvalues(zfArray,BArray):
    zfs1 = zfArray[0]*np.array([[0,.3333,.3333],\
                              [.3333,0,-.3333],\
                              [.3333,-.3333,0]])
    strain1 = zfArray[1]*np.array([[.3333,-.3333,.6667],\
                              [-.3333,-.6667,.3333],\
                              [.6667,.3333,.3333]])
    zfs2 = zfArray[0]*np.array([[0,.3333j,-.3333],\
                              [-.3333j,0,-.3333j],\
                              [-.3333,.3333j,0]])
    strain2 = zfArray[1]*np.array([[.3333,-.3333j,-.6667],\
                              [.3333j,-.6667,.3333j],\
                              [-.6667,-.3333j,.3333]])
    zfs3 = zfArray[0]*np.array([[0,-.3333,.3333],\
                       [-.3333,0,.3333],\
                       [.3333,.3333,0]])
    strain3 = zfArray[1]*np.array([[.3333,.3333,.6667],\
                         [.3333,-.6667,-.3333],\
                         [.6667,-.3333,.3333]])
    zfs4 = zfArray[0]*np.array([[0,-.3333j,-.3333],\
                       [.3333j,0,.3333j],\
                       [-.3333,-.3333j,0]])
    strain4 = zfArray[1]*np.array([[.3333,.3333j,-.6667],\
                          [-.3333j,-.6667,-.3333j],\
                          [-.6667,.3333j,.3333]])
    sx = np.array([[0,.7071,0],[.7071,0,.7071],[0,.7071,0]])
    sy = np.array([[0,-.7071j,0],[-.7071j,0,-.7071j],[0,-.7071j,0]])
    sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])

    # If starting with spherical coordinates
##    zeeman = 28024951642*btpArray[0]*(np.sin(BArray[1])*np.cos(BArray[2])*sx+\
##                            np.sin(BArray[1])*np.sin(BArray[2])*sy+\
##                            np.cos(BArray[2])*sz)

    # If starting with cartesian coordinates
    zeeman = 28024951642 * ((BArray[0]*sx) + \
                           (BArray[1]*sy) + \
                           (BArray[2]*sz))
    w, v = LA.eigh(zfs1+strain1+zeeman)
    f1 = abs(w[0])+abs(w[1])
    f2 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs2+strain2+zeeman)
    f3 = abs(w[0])+abs(w[1])
    f4 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs3+strain3+zeeman)
    f5 = abs(w[0])+abs(w[1])
    f6 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs4+strain4+zeeman)
    f7 = abs(w[0])+abs(w[1])
    f8 = abs(w[0])+abs(w[2])
    evals = np.sort(np.array([f1,f2,f3,f4,f5,f6,f7,f8]))
##    evArray[0] = evals[0]
##    evArray[1] = evals[1]
##    evArray[2] = evals[2]
##    evArray[3] = evals[3]
##    evArray[4] = evals[4]
##    evArray[5] = evals[5]
##    evArray[6] = evals[6]
##    evArray[7] = evals[7]
    return evals
















'''
def eigenvalues(fitArray):
    zfs1 = fitArray[0]*np.array([[0,.3333,.3333],\
                              [.3333,0,-.3333],\
                              [.3333,-.3333,0]])
    strain1 = fitArray[1]*np.array([[.3333,-.3333,.6667],\
                              [-.3333,-.6667,.3333],\
                              [.6667,.3333,.3333]])
    zfs2 = fitArray[0]*np.array([[0,.3333j,-.3333],\
                              [-.3333j,0,-.3333j],\
                              [-.3333,.3333j,0]])
    strain2 = fitArray[1]*np.array([[.3333,-.3333j,-.6667],\
                              [.3333j,-.6667,.3333j],\
                              [-.6667,-.3333j,.3333]])
    zfs3 = fitArray[0]*np.array([[0,-.3333,.3333],\
                       [-.3333,0,.3333],\
                       [.3333,.3333,0]])
    strain3 = fitArray[1]*np.array([[.3333,.3333,.6667],\
                         [.3333,-.6667,-.3333],\
                         [.6667,-.3333,.3333]])
    zfs4 = fitArray[0]*np.array([[0,-.3333j,-.3333],\
                       [.3333j,0,.3333j],\
                       [-.3333,-.3333j,0]])
    strain4 = fitArray[1]*np.array([[.3333,.3333j,-.6667],\
                          [-.3333j,-.6667,-.3333j],\
                          [-.6667,.3333j,.3333]])
    sx = np.array([[0,.7071,0],[.7071,0,.7071],[0,.7071,0]])
    sy = np.array([[0,-.7071j,0],[-.7071j,0,-.7071j],[0,-.7071j,0]])
    sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
    zeeman = 28024951642*fitArray[4]*(np.sin(fitArray[5])*np.cos(fitArray[6])*sx+\
                            np.sin(fitArray[5])*np.sin(fitArray[6])*sy+\
                            np.cos(fitArray[6])*sz)
    w, v = LA.eigh(zfs1+strain1+zeeman)
    f1 = abs(w[0])+abs(w[1])
    f2 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs2+strain2+zeeman)
    f3 = abs(w[0])+abs(w[1])
    f4 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs3+strain3+zeeman)
    f5 = abs(w[0])+abs(w[1])
    f6 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs4+strain4+zeeman)
    f7 = abs(w[0])+abs(w[1])
    f8 = abs(w[0])+abs(w[2])
    evals = np.sort(np.array([f1,f2,f3,f4,f5,f6,f7,f8]))
    fitArray[15] = evals[0]
    fitArray[16] = evals[1]
    fitArray[17] = evals[2]
    fitArray[18] = evals[3]
    fitArray[19] = evals[4]
    fitArray[20] = evals[5]
    fitArray[21] = evals[6]
    fitArray[22] = evals[7]
    return fitArray
##
##fitArray = np.array([2.87082355e9,4.64865979e6,3.90749151e6,-1.27720918e-2,\
##                     6.95e-3,1.4,1.4,\
##                     5e6,5e6,5e6,5e6,5e6,5e6,5e6,5e6,\
##                     0,0,0,0,0,0,0,0])
##print eigenvalues(fitArray)
'''
