import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self):
        #self.T = 10000
        #self.lr = 0.5

        self.sizes = []

        self.nI = 2  # input
        self.nJ = 3  # hidden
        self.nK = 1  # output
        self.nQ = 2

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
            self.J = self.getActivation(self.nJ, self.nI, input, self.WIJ)
            self.Js = np.hstack([sigmoids(self.J),1])

            self.Q = self.getActivation(self.nQ, self.nJ, self.Js, self.WJQ)
            self.Qs = np.hstack([sigmoids(self.Q), 1])

            self.K = self.getActivation(self.nK, self.nQ, self.Qs, self.WQK)
            self.Ks = sigmoids(self.K)


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


#running
Inputs = [[0.0, 0.0, 1], [0.0, 1.0, 1], [1.0, 0.0, 1], [1.0, 1.0, 1]]
Targets = [[0.0], [1.0], [1.0], [0.0]]

M = MultiLayerPerceptron()
T = 10000

for t in range(T):
    r = int(np.floor(np.random.rand() * 4))
    input = Inputs[r]
    target = Targets[r]
    M.update(0.5, input, target)

for i in range(4):
    M.update(0.0, Inputs[i], Targets[i])
    print("output", M.Ks)

plt.plot(M.Errors)
plt.title('Cost over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()











