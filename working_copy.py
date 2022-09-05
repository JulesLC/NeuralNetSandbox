import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import MLP as mlp


#running
#Load Data
D1 = np.load('img1.npz')
X1 = D1['X']
n1 = X1.shape[0]
x1 = X1[:,0]
y1 = X1[:,1]
maxdim1 = np.max([np.max(x1),np.max(y1)])
x1 = x1/maxdim1 # TODO: make sure this is proportionally resizing the images, double check that its staying the same across exported sizes
y1 = y1/maxdim1 #

D2 = np.load('img2.npz')
X2 = D2['X']
n2 = X2.shape[0]
x2 = X2[:,0]
y2 = X2[:,1]
maxdim2 = np.max([np.max(x2),np.max(y2)])
x2 = x2/maxdim2
y2 = y2/maxdim2 #

#print(X)

#Inputs = [[0.0, 0.0, 1], [0.0, 1.0, 1], [1.0, 0.0, 1], [1.0, 1.0, 1]]
#Targets = [[0.0], [1.0], [1.0], [0.0]]

M = mlp.MultiLayerPerceptron(2,15,15,3)
T = 100000
#T = 100

for t in range(T):

    context = (np.random.rand()<0.5)
    if(context):
        r = int(np.floor(np.random.rand() * n1))
        input = [x1[r],y1[r]]
        areaFlag = X1[r,2]

    else:
        r = int(np.floor(np.random.rand() * n2))
        input = [x2[r],y2[r]]
        areaFlag = X2[r,2]

    target = np.zeros(M.nK)*1.0
    if (areaFlag == 1):
        target[2] = 1.0
    elif(areaFlag ==2):
        target[0] = 1.0
    elif (areaFlag == 3):
        target[1] = 1.0
    if((t%100)==0):
        print("progress: ", t/T)

    if (context):
        M.update(0.2, input, target, [0])
    else:
        M.update(0.2, input, target, [])

print('Finished training...')

print('testing map 1...')
R1 = np.zeros([1,M.nK])
for i in range(n1):
    input = np.hstack([x1[i],y1[i]])
    M.update(0.0, input, np.zeros(M.nK),[1,2]) #passing in the index of the positions that get set to zero (disabled)
    R1 = np.vstack([R1, M.Ks])

print('testing map 2...')
R2 = np.zeros([1,M.nK])
for i in range(n2):
    input = np.hstack([x2[i],y2[i]])
    M.update(0.0, input, np.zeros(M.nK),[0,2]) #0,1
    R2 = np.vstack([R2, M.Ks])

print('Finished testing...')

np.savez('testrun.npz',x1=x1,y1=y1,x2=x2,y2=y2,R1=R1,R2=R2,errs=M.Errors)










