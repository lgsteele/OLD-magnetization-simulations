'''
Using Python Version 2.7.12
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def magneticFieldCalc(x1,x2,y1,y2,z1,z2,\
                      X1,X2,Y1,Y2,Z1,Z2,\
                      xsteps,ysteps,zsteps,\
                      Xsteps,Ysteps,Zsteps,\
                      Mx,My,Mz):
    ##############
    # Generate cubic sample volume containing magnetization
    # Note: np.zeroes([rows,columns])
    # Note: form combination array using list comprehension:
    #       https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    ##############
    #Convert input variables to floats so as not to run
    #into any division by zero / integer rounding errors
    x1,x2,y1,y2,z1,z2,\
    X1,X2,Y1,Y2,Z1,Z2,\
    xsteps,ysteps,zsteps,\
    Xsteps,Ysteps,Zsteps,\
    Mx,My,Mz = \
        float(x1),float(x2),float(y1),float(y2),float(z1),float(z2),\
        float(X1),float(X2),float(Y1),float(Y2),float(Z1),float(Z2),\
        float(xsteps),float(ysteps),float(zsteps),\
        float(Xsteps),float(Ysteps),float(Zsteps),\
        float(Mx),float(My),float(Mz)

    #Generate volume of cube containing uniform magnetization
    #use x2+xstep/2 to include x1 and x2 in array
    xstep = (x2-x1)/xsteps
    xArray = np.arange(x1,x2+xstep/2,xstep)
    ystep = (y2-y1)/ysteps
    yArray = np.arange(y1,y2+ystep/2,ystep)
    zstep = (z2-z1)/zsteps
    zArray = np.arange(z1,z2+zstep/2,zstep)
    xyzArray = np.array([[x0,y0,z0] for x0 in xArray \
                                          for y0 in yArray \
                                          for z0 in zArray])
##    print xyzArray

    #Generate volume of cube to assess magnetic field
    #use x2+xstep/2 to include x1 and x2 in array
    #10um laser spot size
    Xstep = (X2-X1)/Xsteps
    XArray = np.arange(X1,X2+Xstep/2,Xstep)
    Ystep = (Y2-Y1)/Ysteps
    YArray = np.arange(Y1,Y2+Ystep/2,Ystep)
    Zstep = (Z2-Z1)/Zsteps
    ZArray = np.arange(Z1,Z2+Zstep/2,Zstep)
    XYZArray = np.array([[x0,y0,z0] for x0 in XArray \
                                       for y0 in YArray \
                                       for z0 in ZArray])
##    print XYZArray
    
    #Generate magnetic field array;
    #Equal in size to XYZArray
    BArray = np.zeros([len(XYZArray),3])
    
    #Generate magnetization and magnetic moment vectors
    #Magnetic moment per atom of Co (bulk): mu = 1.71*uB/atom
    #Density of Co: rho = 8.90 g/cm3
    #Molar mass of Co: MM = 58.9332 g/mol
    #Avogadro's Constant: N = 6.022e23 atoms/mol
    #Magnetic moment of volume of Co: [(mu * N * rho) / (MM)] * dV
    mu = 1.71 * (9.274*(10**(-24))) # J / (T atom)
    rho = 8900. # kg / m3
    molmass = 58.9332 # g / mol
    Navo = 6.022*(10**23) # atoms / mol
    M = (mu * Navo * rho) / molmass
##    Marray = np.array([0,0,M])
##    Marray = np.array([0,M,0])
##    Marray = np.array([0,M/np.sqrt(2),M/np.sqrt(2)])
    Marray = np.array([M/np.sqrt(3),M/np.sqrt(3),M/np.sqrt(3)])
    dVx = xstep
    dVy = ystep
    dVz = zstep
    dV = dVx*dVy*dVz
    m = Marray * dV
    
    
    #Dipole field calculation
    def Bdip(a,b):
        mu0 = 4*np.pi*(10**(-7)) # T m / A
        r = (XYZArray[a,:]-xyzArray[b,:])
        rmag = np.linalg.norm(r)
        rhat = r/(rmag)
        return (mu0/(4*np.pi))*(1/(rmag**3))*(3*np.inner(m,rhat)*rhat-m)

    for i in range(0,len(xyzArray),1): # i is the magnetized sample volume
        for j in range(0,len(XYZArray),1): # j is the volume of interest
##            if xyzArray[i,0] == XYZArray[j,0] and \
##               xyzArray[i,1] == XYZArray[j,1] and \
##               xyzArray[i,2] == XYZArray[j,2]:
##                pass
##            else:
##            print 'mu = ' + str(4*np.pi*(10**(-7)))
            BArray[j,:] = BArray[j,:]+Bdip(j,i)

##            print 'xyzArray = ' + str(xyzArray[i,:])
##            print 'XYZArray = ' + str(XYZArray[j,:])
##            r = (XYZArray[j,:]-xyzArray[i,:])
##            rmag = np.linalg.norm(r)
##            rhat = r/(rmag)
##            print 'r = ' + str(r)
##            print 'rmag = ' + str(rmag)
##            print 'rhat = ' + str(rhat)
##            print 'r3 = ' + str(rmag**3) + '\n'
##            
##            print 'm = ' + str(m)
##            print 'rhat = ' + str(rhat)
##            print 'np.inner(m,rhat) = ' + str(np.inner(m,rhat))
##            print '3*np.inner(m,rhat)*rhat = ' + \
##                  str(3*np.inner(m,rhat)*rhat)
##            print '3*np.inner(m,rhat)*rhat - m = ' + \
##                  str(3*np.inner(m,rhat)*rhat - m)
##            print 'Bdip = ' + str(Bdip(j,i))
##            print BArray
            

    #Calculate length of magnetic field vectors
    #for use in plotting
##    BampArray = np.zeros([len(BArray),1])
##    for i in range(0,len(BampArray),1):
##        BampArray[i] = np.linalg.norm(BArray[i,:])*(10**5)
##    BArray = np.column_stack((BArray, BampArray))
##    print BArray

    BNormArray = BArray/(10000*np.amax(BArray))
##    print BNormArray
    vectorfieldplotting = np.column_stack((XYZArray,BNormArray))
    vectorfield = np.column_stack((XYZArray,BArray))
##    print vectorfield

    # Calculate avg/stderr of Bz in region of interest
    print 'Bx avg = ' + str(np.average(vectorfield[:,3]))
    print 'Bx stdev = ' + str(np.std(vectorfield[:,3]))
    print 'By avg = ' + str(np.average(vectorfield[:,4]))
    print 'By stdev = ' + str(np.std(vectorfield[:,4]))
    print 'Bz average = ' + str(np.average(vectorfield[:,5]))
    print 'Bz stdev = ' + str(np.std(vectorfield[:,5]))
    
    try:
        x,y,z,u,v,w = zip(*vectorfieldplotting)
        fig = plt.figure()
    ##    ax = fig.gca(projection='3d')
    ##    ax = plt.gca()
        ax = fig.add_subplot(111,projection='3d')
    ##    x, y, z = np.meshgrid(np.arange(-1, 2, 0.5),
    ##                      np.arange(-1, 2, 0.5),
    ##                      np.arange(-1, 2, 0.5))
    ##    ax.quiver(vectorfield[:,0],vectorfield[:,1],vectorfield[:,2],\
    ##              vectorfield[:,3],vectorfield[:,4],vectorfield[:,5])
##        x = np.array([x1,x2])
##        ax.plot(x,y1,)
        ax.quiver(x,y,z,u,v,w,pivot='middle',normalize=False)
##        ax.set_xlim(-.1,3.1,.1)
##        ax.set_ylim(-.1,3.1,.1)
##        ax.set_zlim(-.1,3.1,.1)
        plt.draw()
        plt.show()
    except KeyboardInterrupt:
        plt.ioff()
        plt.close()
        os._exit(0)

    





##magneticFieldCalc(x1,x2,y1,y2,z1,z2,\
##                      X1,X2,Y1,Y2,Z1,Z2,\
##                      xsteps,ysteps,zsteps\
##                      Xsteps,Ysteps,Zsteps,\
##                      Mx,My,Mz):
# Mz
magneticFieldCalc(.0012,.0018,.0012,.0018,.002,.0026,\
                  .0014,.0016,.0014,.0016,.0012,.0016,\
                  5,5,5,\
                  5,5,5,\
                  0,0,1)
##magneticFieldCalc(1,2,1,2,2,3,\
##                  1,2,1,2,1,2,\
##                  5,5,5,\
##                  5,5,5,\
##                  0,0,1)





##m = np.array([0,0,1])
##xyztmp = np.array([0,0,0])
##XYZtmp = np.array([0,0,.001])
##print XYZtmp-xyztmp
##def Bdiptest():
##    mu0 = 4*np.pi*10**(-7) # T m / A
##    r = (XYZtmp-xyztmp)
##    print r
##    rhat = r/(np.linalg.norm(XYZtmp-xyztmp))
##    print rhat
##    print (mu0/(4*np.pi))*(1/np.linalg.norm(r)**3)*(3*m.dot(rhat)*rhat-m)
##Bdiptest()
