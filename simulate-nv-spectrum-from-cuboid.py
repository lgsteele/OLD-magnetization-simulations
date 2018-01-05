import numpy as np
import matplotlib.pyplot as plt

from magnetic_field_outside_cuboid import magneticFieldCalc
from NVeigenvalues import eigenvalues
from lor8 import lor8



magneticFieldCalc(.0011,.0019,.0011,.0019,.002,.0028,\
                  .0015,.0017,.0014,.0016,.0014,.0016,\
                  5,5,5,\
                  5,5,5,\
                  0,0,1)

#Upload Bavg data from magneticFieldCalc
Bxavg, Byavg, Bzavg, Bxstd, Bystd, Bzstd = \
       np.loadtxt('BavgArray.txt', delimiter = ', ', unpack=True)
BxyzArray = np.array([Bxavg, Byavg, Bzavg, Bxstd, Bystd, Bzstd])

#Calculate eigenvalues given BavgArray from simulation
#Note: need to define zfArray here
'''
Original Co data set:
[data] = [D, E, width]
zero field: 2.87078235e+09, 2.97186918e+06, 2.27586711e+06
Co: 2.87086163e+09, 4.41376381e+06, 3.33958353e+06
Size of Co sample: unknown

Given zfArray = zero field, (simulatedEV[6] - simulatedEV[1])/2 should equal ~4.4e6
To properly confirm this, need to include Bstd in error
'''
zfArray = np.array([2.87078235e+09, 2.97186918e+06, 2.27586711e+06,0])
simulatedEV = eigenvalues(zfArray, BxyzArray)
#Calculate eigenvalues for zero-field
zeroB = np.array([0,0,0,0,0,0])
zfEV = eigenvalues(zfArray, zeroB)
#Print eigenvalues for reference
print 'ZF: ' + str(zfEV)
print 'simulated: ' + str(simulatedEV)
print (simulatedEV[6] - simulatedEV[1])/2
print (zfEV[6] - zfEV[1])/2



#Generate spectra
freq = np.arange(2.77e9,2.97e9,1e6)
ampArray = np.array([1,1,1,1,1,1,1,1])
simulatedSpectra = lor8(freq,zfArray,ampArray,simulatedEV)
zfSpectra = lor8(freq,zfArray,ampArray,zfEV)



#Plot results
try:
    plt.plot(freq,simulatedSpectra,'r',freq,zfSpectra,'b-')
    plt.show()
except KeyboardInterrupt:
    plt.ioff()
    plt.close()
    os._exit(0)
