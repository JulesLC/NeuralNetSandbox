import numpy as np
import h5py
import pylab as pl
from scipy.interpolate import interp1d

x = h5py.File('/Users/julialc4/Desktop/MLP/b2/logs/out.h5', 'r')

Error = x['errors'][:] #get everything inside the labelled h5 key'/error'
print(Error)

#first image
OutNode1 = x['output1'][:]
OutNode2 = x['output2'][:]
OutNode3 = x['output3'][:]

x1 = x['x1'][:]
y1 = x['y1'][:]

Y = np.ones([int(np.max(x1))+1, int(np.max(y1))+1, 3])

for i in range(len(x1)):
    Y[int(x1[i]), int(y1[i]), 2] = OutNode1[i]> 0.5
    Y[int(x1[i]), int(y1[i]), 0] = OutNode2[i]>0.5
    Y[int(x1[i]), int(y1[i]), 1] = OutNode3[i]>0.5 #filters out otherwise is raw

print(Y.shape)

Y = np.transpose(Y, [1,0,2]) #swap x, y
print(Y.shape)


print("x1 size", len(x1))
print("y1 size", len(y1))
print("node 1", len(OutNode1))


#second image
OutNodeA1 = x['outputA1'][:]
OutNodeA2 = x['outputA2'][:]
OutNodeA3 = x['outputA3'][:]

x2 = x['x2'][:]
y2 = x['y2'][:]

Y2 = np.ones([int(np.max(x2))+1, int(np.max(y2))+1, 3])

for i in range(len(x2)):
    Y2[int(x2[i]), int(y2[i]), 2] = OutNodeA1[i]> 0.5
    Y2[int(x2[i]), int(y2[i]), 0] = OutNodeA2[i]>0.5
    Y2[int(x2[i]), int(y2[i]), 1] = OutNodeA3[i]>0.5 #filters out otherwise is raw

Y2 = np.transpose(Y2, [1,0,2]) #swap x, y

#interp image
OutNodeB1 = x['outputB1'][:]
OutNodeB2 = x['outputB2'][:]
OutNodeB3 = x['outputB3'][:]

print("x1 shape", x1.shape)
print("x2 shape", x2.shape)

#x3 = interp1d(x1, x2)
#y3 = interp1d(y1, y2)
x3 = x2
y3 = y2

Y3 = np.ones([int(np.max(x3))+1, int(np.max(y3))+1, 3])

for i in range(len(x3)):
    Y3[int(x3[i]), int(y3[i]), 2] = OutNodeB1[i]> 0.5
    Y3[int(x3[i]), int(y3[i]), 0] = OutNodeB2[i]>0.5
    Y3[int(x3[i]), int(y3[i]), 1] = OutNodeB3[i]>0.5 #filters out otherwise is raw

Y3 = np.transpose(Y3, [1,0,2]) #swap x, y


F = pl.figure()
f = F.add_subplot(221)
f.set_xlabel('Errors')
f.plot(Error)
f = F.add_subplot(222)
f.set_xlabel('1st map')
f.imshow(Y)
f = F.add_subplot(223)
f.set_xlabel('2nd map')
f.imshow(Y2)
f = F.add_subplot(224)
f.set_xlabel('interp map')
f.imshow(Y3)

pl.show()


