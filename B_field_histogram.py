'''
Generate histogram of Bx, By, Bz
'''

import numpy as np
import matplotlib.pyplot as plt

from magnetic_field_outside_uniformly_magnetized_cuboid import magneticFieldCalc
from NVeigenvalues import eigenvalues
from lor8 import lor8



try:
    magneticFieldCalc(.0011,.0019,.0011,.0019,.002,.0028,\
                  .0014,.0016,.0014,.0016,.0014,.0016,\
                  5,5,5,\
                  5,5,5,\
                  1,0,0)
    vectorfield = \
       np.loadtxt('vectorfield.txt', delimiter = ', ', unpack=False)
##    vectorfield = vectorfield*(10**4)
    x,y,z,Bx,By,Bz = zip(*vectorfield)
    fig = plt.figure(figsize=plt.figaspect(1.))
    
    ax = fig.add_subplot(331)
    ax = plt.hist(vectorfield[:,3],bins=100,density=False)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,3])),\
                           np.amax(np.absolute(vectorfield[:,3])))
    ax = plt.title('M = (1,0,0)')
    
    ax = fig.add_subplot(334)
    ax = plt.hist(vectorfield[:,4],bins=100)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,4])),\
                           np.amax(np.absolute(vectorfield[:,4])))
    
    ax = fig.add_subplot(337)
    ax = plt.hist(vectorfield[:,5],bins=100)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,5])),\
                           np.amax(np.absolute(vectorfield[:,5])))

    
    magneticFieldCalc(.0011,.0019,.0011,.0019,.002,.0028,\
                  .0014,.0016,.0014,.0016,.0014,.0016,\
                  5,5,5,\
                  5,5,5,\
                  0,1,0)
    vectorfield = \
       np.loadtxt('vectorfield.txt', delimiter = ', ', unpack=False)
##    vectorfield = vectorfield*(10**4)
    x,y,z,Bx,By,Bz = zip(*vectorfield)
    ax = fig.add_subplot(332)
    ax = plt.hist(vectorfield[:,3],bins=100,density=False)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,3])),\
                           np.amax(np.absolute(vectorfield[:,3])))
    ax = plt.title('M = (0,1,0)')
    
    ax = fig.add_subplot(335)
    ax = plt.hist(vectorfield[:,4],bins=100)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,4])),\
                           np.amax(np.absolute(vectorfield[:,4])))
    
    ax = fig.add_subplot(338)
    ax = plt.hist(vectorfield[:,5],bins=100)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,5])),\
                           np.amax(np.absolute(vectorfield[:,5])))


    magneticFieldCalc(.0011,.0019,.0011,.0019,.002,.0028,\
                  .0014,.0016,.0014,.0016,.0014,.0016,\
                  5,5,5,\
                  5,5,5,\
                  0,0,1)
    vectorfield = \
       np.loadtxt('vectorfield.txt', delimiter = ', ', unpack=False)
##    vectorfield = vectorfield*(10**4)
    x,y,z,Bx,By,Bz = zip(*vectorfield)
    ax = fig.add_subplot(333)
    ax = plt.hist(vectorfield[:,3],bins=100,density=False)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,3])),\
                           np.amax(np.absolute(vectorfield[:,3])))
    ax = plt.title('M = (0,0,1)')
    
    ax = fig.add_subplot(336)
    ax = plt.hist(vectorfield[:,4],bins=100)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,4])),\
                           np.amax(np.absolute(vectorfield[:,4])))
    
    ax = fig.add_subplot(339)
    ax = plt.hist(vectorfield[:,5],bins=100)
    ax = plt.xlim(-np.amax(np.absolute(vectorfield[:,5])),\
                           np.amax(np.absolute(vectorfield[:,5])))
    
    plt.draw()
    plt.show()
except KeyboardInterrupt:
    plt.ioff()
    plt.close()
    os._exit(0)
