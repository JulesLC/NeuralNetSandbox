import numpy as np
import pylab as pl
D = np.load('testrun.npz')
x=D['x']
y=D['y']
R=D['R'] #R is response
errs=D['errs']
n = len(x)
print(np.min(x))
print(np.max(x))
print(np.min(y))
print(np.max(y))

Rrounded = (R>0.5)*1

print(Rrounded.shape)

D = np.load('img1.npz')


F = pl.figure()
f = F.add_subplot(121)
f.plot(errs)

f = F.add_subplot(122)
j = np.random.permutation(n)

for i in range(5000): # n is all
    f.plot(x[j[i]], -y[j[i]], '.', color=Rrounded[j[i]])

f.set_aspect(np.diff(f.get_xlim())/np.diff(f.get_ylim()))
'''
for i in range(n):
    if(R[i]>0.5):
        f.plot(x[i],y[i],'.',color='k')
'''

pl.show()
