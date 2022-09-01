import numpy as np
import pylab as pl
D = np.load('testrun.npz')
x=D['x']
y=D['y']
R=D['R']
errs=D['errs']
n = len(x)

F = pl.figure()
f = F.add_subplot(121)
f.plot(errs)

f = F.add_subplot(122)
for i in range(n):
    if(R[i]>0.51):
        f.plot(x[i],1-y[i],'.',markersize=0.5,color='b')
f.axis([0,+1,0,+1])

pl.show()

