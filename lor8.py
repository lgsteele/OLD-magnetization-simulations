import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import peakutils

def lor8(freq,zfArray,ampArray,evArray):
    width = zfArray[2]
    offset = zfArray[3]
    def func(freq,ampArray,evArray):
        lor1 = ampArray[0]*((width**2)/(np.pi*width*((freq-evArray[0])**2 + width**2)))
        lor2 = ampArray[1]*((width**2)/(np.pi*width*((freq-evArray[1])**2 + width**2)))
        lor3 = ampArray[2]*((width**2)/(np.pi*width*((freq-evArray[2])**2 + width**2)))
        lor4 = ampArray[3]*((width**2)/(np.pi*width*((freq-evArray[3])**2 + width**2)))
        lor5 = ampArray[4]*((width**2)/(np.pi*width*((freq-evArray[4])**2 + width**2)))
        lor6 = ampArray[5]*((width**2)/(np.pi*width*((freq-evArray[5])**2 + width**2)))
        lor7 = ampArray[6]*((width**2)/(np.pi*width*((freq-evArray[6])**2 + width**2)))
        lor8 = ampArray[7]*((width**2)/(np.pi*width*((freq-evArray[7])**2 + width**2)))
        return lor1 + lor2 + lor3 + lor4 + lor5 + lor6 + lor7 + lor8 + offset
    return func(freq,ampArray,evArray)
