import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class MultiLayerPerceptron():
    def __init__(self):
        #self.T = 10000
        #self.lr = 0.5

        self.nI = 2  # input
        self.nJ = 3  # hidden
        self.nK = 1  # output

        self.WIJ = np.random.rand(self.nI + 1, self.nJ) * 2 - 1
        self.WJK = np.random.rand(self.nJ + 1, self.nK) * 2 - 1

        self.Errors = []

        self.J = np.ones([self.nJ])  # nj, 1
        self.Js = np.ones([self.nJ + 1])
        self.K = np.ones([self.nK])
        self.Ks = np.ones([self.nK])


    def update(self, lr, input, target):
            # forward pass

            for j in range(self.nJ):
                self.J[j] = 0
                for i in range(self.nI + 1):
                    self.J[j] += input[i] * self.WIJ[i, j]
                self.Js[j] = sigmoid(self.J[j])

            for k in range(self.nK):
                self.K[k] = 0
                for j in range(self.nJ + 1):
                    self.K[k] += self.Js[j] * self.WJK[j, k]
                self.Ks[k] = sigmoid(self.K[k])

            # backward pass
            deltasK = np.zeros(self.nK)
            for k in range(self.nK):
                deltasK[k] = -(target[k] - self.Ks[k])

            deltasJ = np.zeros(self.nJ)
            for j in range(self.nJ):
                for k in range(self.nK):
                    deltasJ[j] += deltasK[k] * self.WJK[j, k]
            for j in range(self.nJ):
                deltasJ[j] *= sigmoid_prime(self.J[j])

            # update weight/biases
            for j in range(self.nJ + 1):
                for k in range(self.nK):
                    self.WJK[j, k] += -(self.Js[j] * deltasK[k] * lr)

            for i in range(self.nI + 1):
                for j in range(self.nJ):
                    self.WIJ[i, j] += -(input[i] * deltasJ[j] * lr)

            # track errors
            Error = np.sum(deltasK ** 2)
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











