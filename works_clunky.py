import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

T = 10000
lr = 0.5

trainingInput = [[0.0, 0.0, 1], [0.0, 1.0, 1], [1.0, 0.0, 1], [1.0, 1.0, 1]]
trainingOutput = [[0.0], [1.0], [1.0], [0.0]]

nI = 2  #input
nJ = 3  #hidden
nK = 1  #output

WIJ = np.random.rand(nI+1, nJ) * 2-1
WJK = np.random.rand(nJ+1, nK) * 2-1

Errors = np.zeros(T)

for t in range(T):
    r = int(np.floor(np.random.rand() * 4))
    input = trainingInput[r]
    target = trainingOutput[r]
    #forward pass
    J = np.ones([nJ]) #nj, 1
    Js = np.ones([nJ+1])
    for j in range(nJ):
        J[j] = 0
        for i in range(nI+1):
            J[j] += input[i] * WIJ[i, j]
        Js[j] = sigmoid(J[j])
    K = np.ones([nK])
    Ks = np.ones([nK])
    for k in range(nK):
        K[k] = 0
        for j in range(nJ+1):
            K[k] += Js[j] * WJK[j, k]
        Ks[k] = sigmoid(K[k])

    #backward pass
    deltasK = np.zeros(nK)
    for k in range(nK):
        deltasK[k] = -(target[k]-Ks[k])

    deltasJ = np.zeros(nJ)
    for j in range(nJ):
        for k in range(nK):
            deltasJ[j] += deltasK[k] * WJK[j, k]
    for j in range(nJ):
        deltasJ[j] *= sigmoid_prime(J[j])

    #update weight/biases
    for j in range(nJ+1):
        for k in range(nK):
            WJK[j, k] += -(Js[j] * deltasK[k] * lr)

    for i in range(nI+1):
        for j in range(nJ):
            WIJ[i, j] += -(input[i] * deltasJ[j] * lr)

    #track errors
    Errors[t] = np.sum(deltasK**2)

plt.plot(Errors)
plt.title('Cost over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()









