'''
Using Python Version 2.7.12

Notes/References:
    np.zeros([rows,columns])
    form combination array using list comprehension:
    https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob

freq = np.arange(0.87e9,4.87e9,1e6)

sum_2 = np.loadtxt('sum_2')
sum_4 = np.loadtxt('sum_4')
sum_8 = np.loadtxt('sum_8')
sum_12 = np.loadtxt('sum_12.txt')
sum_16 = np.loadtxt('sum_16')
sum_24 = np.loadtxt('sum_24.txt')
sum_32 = np.loadtxt('sum_32')
print len(sum_2)
print len(sum_12)
print len(freq)

sum_2 = sum_2/np.amax(sum_2)
sum_4 = sum_4/np.amax(sum_4)
sum_8 = sum_8/np.amax(sum_8)
sum_12 = sum_12/np.amax(sum_12)
sum_16 = sum_16/np.amax(sum_16)
sum_24 = sum_24/np.amax(sum_24)
sum_32 = sum_32/np.amax(sum_32)

plt.plot(freq,sum_2+2, freq,sum_4+4, freq,sum_8+8, \
         freq,sum_12+12, freq,sum_16+16, freq,sum_24+24, \
         freq,sum_32+32)
plt.show()

for i in [2,4,8,16,32]:
    pass
