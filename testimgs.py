import numpy as np
import pylab as pl
D = np.load('testrun.npz')
x1=D['x1']
y1=D['y1']
R1=D['R1'] #R is response

x2=D['x2']
y2=D['y2']
R2=D['R2'] #R is response


errs=D['errs']
n1 = len(x1)
n2 = len(x2)

Rrounded1 = (R1>0.5)*1
Rrounded2 = (R2>0.5)*1



F = pl.figure()
f = F.add_subplot(131)
f.plot(errs)

f = F.add_subplot(132)
j = np.random.permutation(n1)
for i in range(5000): # n is all
    f.plot(x1[j[i]], -y1[j[i]], '.', markersize=2,color=Rrounded1[j[i]])

f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))

f = F.add_subplot(133)
j = np.random.permutation(n2)
for i in range(5000): # n is all
    f.plot(x2[j[i]], -y2[j[i]], '.', markersize=2, color=Rrounded2[j[i]])

f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))

pl.show()
