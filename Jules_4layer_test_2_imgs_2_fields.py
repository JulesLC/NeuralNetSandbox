import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import MLP as mlp

'''
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def sigmoids(z):
    """The sigmoid function."""
    zs = np.zeros(len(z))
    for i in range(len(z)):
        zs[i] = sigmoid(z[i])
    return zs

class MultiLayerPerceptron():
    def __init__(self, nI,nJ,nQ,nK):
        #self.T = 10000
        #self.lr = 0.5

        self.sizes = []

        self.nI = nI # input
        self.nJ = nJ # hidden
        self.nQ = nQ
        self.nK = nK # output

        self.WIJ = np.random.rand(self.nI + 1, self.nJ) * 2 - 1
        self.WJQ = np.random.rand(self.nJ + 1, self.nQ) * 2 - 1
        self.WQK = np.random.rand(self.nQ + 1, self.nK) * 2 - 1

        self.Errors = []

        self.J = np.ones([self.nJ])  # nj, 1
        self.Js = np.ones([self.nJ + 1])
        self.K = np.ones([self.nK])
        self.Ks = np.ones([self.nK])

    def getActivation(self, nOut, nIn, input, weights):
        activation = np.zeros(nOut)
        for j in range(nOut):
            activation[j] = 0
            for i in range(nIn + 1):
                activation[j] += input[i] * weights[i, j]
        return activation

    def getBackward(self, nFromAbove, nHidden, deltasFromAbove, weightsToAbove, activationHidden):
        deltas = np.zeros(nHidden)
        for j in range(nHidden):
            for k in range(nFromAbove):
                deltas[j] += deltasFromAbove[k] * weightsToAbove[j, k]
        for j in range(nHidden):
            deltas[j] *= sigmoid_prime(activationHidden[j])
        return deltas

    def updateWeight(self, nFromAbove, nHidden, weightsToAbove, sigmoidedHiddenActivation,deltasFromAbove, lr):
        for j in range(nHidden + 1):
            for k in range(nFromAbove):
                weightsToAbove[j, k] += -(sigmoidedHiddenActivation[j] * deltasFromAbove[k] * lr)
        return weightsToAbove


    def update(self, lr, input, target):
            # forward pass

            input = np.hstack([input, 1])
            self.J = self.getActivation(self.nJ, self.nI, input, self.WIJ)
            self.Js = np.hstack([sigmoids(self.J),1])

            self.Q = self.getActivation(self.nQ, self.nJ, self.Js, self.WJQ)
            self.Qs = np.hstack([sigmoids(self.Q), 1])

            self.K = self.getActivation(self.nK, self.nQ, self.Qs, self.WQK)
            self.Ks = sigmoids(self.K)

            if(lr>0.0):
                # backward pass from output layer
                outputDeltas = np.zeros(self.nK)
                for k in range(self.nK):
                    outputDeltas[k] = -(target[k] - self.Ks[k])

                deltasQ = self.getBackward(self.nK, self.nQ, outputDeltas, self.WQK, self.Q)

                deltasJ = self.getBackward(self.nQ, self.nJ, deltasQ, self.WJQ, self.J)

                # update weight/biases

                self.WQK = self.updateWeight(self.nK, self.nQ, self.WQK, self.Qs, outputDeltas, lr)

                self.WJQ = self.updateWeight(self.nQ, self.nJ, self.WJQ, self.Js, deltasQ, lr)

                self.WIJ = self.updateWeight(self.nJ, self.nI, self.WIJ, input, deltasJ, lr)

                # track errors
                Error = np.sum(outputDeltas ** 2)
                self.Errors = np.hstack([self.Errors, Error])
'''

#running
#Load Data
D = np.load('img1.npz')
D2 = np.load('img2.npz')
X = D['X']
n = X.shape[0]
x = X[:,0]
y = X[:,1]
maxdim = np.max([np.max(x),np.max(y)])
x = x/maxdim # TODO: make sure this is proportionally resizing the images, double check that its staying the same across exported sizes
y = y/maxdim #

#print(X)

#Inputs = [[0.0, 0.0, 1], [0.0, 1.0, 1], [1.0, 0.0, 1], [1.0, 1.0, 1]]
#Targets = [[0.0], [1.0], [1.0], [0.0]]

M = mlp.MultiLayerPerceptron(2,15,15,3)
T = 1000000
#T = 10

for t in range(T):
    r = int(np.floor(np.random.rand() * n)) #TODO:here we need to add the random draw from one and the other image, each random draw will also set a context node on or off
    input = [x[r],y[r]]
    tar = [X[r,2]]   #TODO:in here is where we add the code for flag 1 being the output of 0, 0, 1 (a1 off, s1 off, v1 on)
    target = [0,0,0]
    if(tar==1):
        target=[1,0,0]
    elif(tar==2):
        target = [0, 1, 0]
    elif (tar == 3):
        target = [0, 0, 1]
    if((t%100)==0):
        print("progress: ", t/T)

    M.update(0.5, input, target)

print('Finished training...')

R = np.zeros([1,3])
for i in range(n):
    input = np.hstack([x[i],y[i]])
    M.update(0.0, input, [0.0,0.0,0.0])
    R = np.vstack([R, M.Ks])

print('Finished testing...')

np.savez('testrun.npz',x=x,y=y,R=R,errs=M.Errors)










