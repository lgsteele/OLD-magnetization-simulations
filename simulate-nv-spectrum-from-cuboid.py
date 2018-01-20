import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime

from magnetic_field_outside_uniformly_magnetized_cuboid import magneticFieldCalc
from NVeigenvalues import eigenvalues
from lor8 import lor8


def simulation(steps,dom):
    print datetime.datetime.now()
    magneticFieldCalc(.0010,.0020,.0020,.0030,.0010,.0020,\
                      .00149,.00151,.00149,.00151,.00149,.00151,\
                      steps,steps,steps,\
                      10,10,10,\
                      0,0,1,
                      dom)

    # Need to generate NV-spectra in each dV and sum signals
    vectorfield = np.loadtxt('vectorfield_%s.txt'%dom, \
                             delimiter = ', ', unpack=False)
    # Need to isolate the magnetic field components of the array
    # (required for NVeigenvalues.py)
    BArray = np.delete(np.copy(vectorfield),(0,1,2),axis=1)
    zfArray = np.array([2.87e9,3e6,2e6,0])
    # there are 8 eigenvalues for each dV
    # add 8 EV columns to vectorfield matrix
    vectorfield = np.column_stack((vectorfield,\
                                   np.zeros((len(vectorfield),8))))
    # Calculate eigenvalues for each dV:
    for i in range(0,len(vectorfield),1):
        simulatedEV = eigenvalues(zfArray,BArray[i,:])
        vectorfield[i,6:14] = simulatedEV
    # Isolate eigenvalue array
    evArray = np.delete(vectorfield,(0,1,2,3,4,5),axis=1)
    # Generate summed spectra for B-field distribution
    freq = np.arange(0.87e9,4.87e9,1e6)
    ampArray = np.array([1e7,1e7,1e7,1e7,1e7,1e7,1e7,1e7])
    spectraSum = np.zeros(len(freq))
    for i in range(0,len(vectorfield),1):
        simulatedSpectra = lor8(freq,zfArray,ampArray,evArray[i,:])
        spectraSum = spectraSum + simulatedSpectra
    np.savetxt('sum_%s.txt'%dom, spectraSum,\
               delimiter = ', ')


    # Generate zerofield spectra
    zeroB = np.array([0,0,0])
    ampArray = ampArray
    zfEV = eigenvalues(zfArray, zeroB)
    zfSpectra = lor8(freq,zfArray,ampArray,zfEV)*len(vectorfield)
    np.savetxt('zf_%s.txt'%dom, zfSpectra,\
               delimiter = ', ')



    ###Upload Bavg data from magneticFieldCalc
    Bxavg, Byavg, Bzavg, Bxstd, Bystd, Bzstd = \
           np.loadtxt('BavgArray_%s.txt'%dom, delimiter = ', ', unpack=True)
    BxyzArray = np.array([Bxavg, Byavg, Bzavg, Bxstd, Bystd, Bzstd])
    ##print BxyzArray

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
    ##zfArray = np.array([2.87078235e+09, 2.97186918e+06, 2.27586711e+06,0])
    simulatedEV = eigenvalues(zfArray, BxyzArray)
    #Calculate eigenvalues for zero-field
    ##zeroB = np.array([0,0,0,0,0,0])
    ##zfEV = eigenvalues(zfArray, zeroB)
    #Print eigenvalues for reference
    ##print 'ZF: ' + str(zfEV) + '\n'
    ##print 'simulated: ' + str(simulatedEV) + '\n'
    ##print (simulatedEV[6] - simulatedEV[1])/2
    ##print (zfEV[6] - zfEV[1])/2



    #Generate spectra
    ##freq = np.arange(2.77e9,2.97e9,1e6)
    ##ampArray = np.array([1e7,1e7,1e7,1e7,1e7,1e7,1e7,1e7])
    simulatedSpectra = lor8(freq,zfArray,ampArray,simulatedEV) * len(vectorfield)
    np.savetxt('avg_%s.txt'%dom, simulatedSpectra,\
               delimiter = ', ')
    ##zfSpectra = lor8(freq,zfArray,ampArray,zfEV)


    print datetime.datetime.now()

    #Plot vectorfield and spectra results
    #First need to upload vectorfield data
    ##vectorfield = np.loadtxt('vectorfield.txt', delimiter = ', ', unpack=False)
    #Next comes the actual plotting
##    try:
##        x,y,z,Bx,By,Bz,e1,e2,e3,e4,e5,e6,e7,e8 = zip(*vectorfield)
##        fig = plt.figure(figsize=plt.figaspect(1.))
##        
##        ax = fig.add_subplot(321,projection='3d')
##        ax.quiver(x,y,z,Bx,By,Bz,pivot='middle',length=.2,normalize=False)
##        
##        ax = fig.add_subplot(323)
##    ##    ax.plot(freq,zfSpectra,'b-',freq,spectraSum,'r-',freq,simulatedSpectra,'g-')
##    ##    ax.annotate('blue = zero field \n red = sum over volumes \n green = avg', xy=(2.88e9,300))
##        ax.plot(freq,zfSpectra,'b-',freq,spectraSum,'r-')
##        ax.annotate('blue = zero field \n red = sum over volumes', \
##                    xy=(2.88e9,2*len(vectorfield)))
##        
##        ax = fig.add_subplot(322)
##        ax = plt.hist(vectorfield[:,3],bins=100,density=False)
##        ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,3])),\
##                               np.amax(np.absolute(vectorfield[:,3])))
##        
##        ax = fig.add_subplot(324)
##        ax = plt.hist(vectorfield[:,4],bins=100)
##        ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,4])),\
##                               np.amax(np.absolute(vectorfield[:,4])))
##        
##        ax = fig.add_subplot(326)
##        ax = plt.hist(vectorfield[:,5],bins=100)
##        ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,5])),\
##                               np.amax(np.absolute(vectorfield[:,5])))
##        
##        plt.draw()
##        plt.show()
##    except KeyboardInterrupt:
##        plt.ioff()
##        plt.close()
##        os._exit(0)






##simulation(12,12,12,0,0,1,12)
##for dom in [20]:
##    simulation(20,20,20,0,0,1,dom)
##for dom in [28]:
##    simulation(28,28,28,0,0,1,dom)

##for dom in [2,4,8,16,32]:
##    simulation(32,32,32,0,1,0,dom)
##for dom in [6,12,24]:
##    simulation(24,24,24,0,1,0,dom)
##for dom in [10,20,30]:
##    simulation(30,30,30,0,1,0,dom)
##for dom in [14,28]:
##    simulation(28,28,28,0,1,0,dom)

##for dom in [2,4,8,16]:
simulation(16,1)
##for dom in [12,24]:
##    simulation(24,dom)










