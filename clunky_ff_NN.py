import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

sizes = [2,3,1]

initA = 0
initB = 1

numberofEpochs = 1000
gain = 3.0 #learningn rate

trainingInput = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
trainingOutput = [[0.0], [1.0], [1.0], [0.0]]

trainingdata = zip(trainingInput, trainingOutput)

#note: the above zip can't be a for loop because its a parrallel iterate
#for x, y in trainingdata:
    #print(x,y)

def visualize(nodes, edges):
    # visualization
    G = nx.Graph()
    #for e in edges:
      #  G.add_edge_value(e.split(","))
    for n in nodes:
        G.add_node(n)
    nx.draw(G, with_labels=True)
    plt.show()


def initiate():
     for layer in sizes:
        for inputlayer in range(sizes[0]):
            inputlayer = np.array([initA, initB])
        for layer1 in range(sizes[1]):
            layer1 = np.random.uniform(0, 1, sizes[1]) #bias term (not the threshold +/- bias), needed unless rand is set to stay positive
        for output in range(sizes[2]):
            output = np.random.uniform(0, 1, sizes[2])
            #outputbiasT = [np.random.randint(-1,1)] #bias threshold version
        net = [inputlayer, layer1, output]



        #input to hidden layer as one array
        #hidden layer to output as second array in one matrix
        weights = [np.random.randn(y, x)
                   for x, y in zip(sizes[:-1], sizes[1:])]
        net.append(weights)
        print("weights ", weights)


        biases = [np.random.randn(y, 1) for y in sizes[1:]]
        print("biases", biases)
        net.append(biases)


        #print(net[2])
        #visualize(net, net[3])
        return net



def feedforward(a, weights, biases):
    # dot product weight of input plus the bias through sigmoid function
    for b, w in zip(biases,weights):
        a = sigmoid(np.dot(w, a) + b)
    return a


def update(biases, weights, trainingdata, gain):
    #train using gradient descent/backpropogate
    #create empty matrices of the same form
    #setting to 0s keeps the form correct (?)
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]

    #here we are populating the new matrices
    for x, y in trainingdata: #x,y is our input to target output mapping
        # delta_nabla_b... will be the gradient of the cost function from back prop
        delta_nabla_b, delta_nabla_w = backpropogate(biases, weights, x, y)
        for b, delta_b in zip(nabla_b, delta_nabla_b): # 0 mapped with gradient result
            nabla_b = [b + delta_b]
        for w, delta_w in zip(nabla_w, delta_nabla_w):
            nabla_w = [w + delta_w]


    # here is where update calculation happens:
    # set the new weights and biases,
    # averaging over the total size and learning rate step size
    for b, new_b in zip(biases, nabla_b):
        biases = [b-(gain/len(trainingdata)) * new_b]

    for w, new_w in zip(weights, nabla_w):
        weights = [w-(gain/len(trainingdata)) * new_w]


def learn(biases, weights, trainingdata, gain):
    #loops over training examples
    for example in trainingdata: #per example update?
        update(biases, weights, trainingdata, gain)
        #TODO: evaluate/visualize performance with each trial


def backpropogate(biases, weights, train_in, train_out):
    #get the gradient of the cost function
    dCdb = [np.zeros(b.shape) for b in biases]
    dCdw = [np.zeros(w.shape) for w in weights]
    print("in", train_in)
    # feed forward pass to get the weighted sum zj
    # inputs used again to seperate out sigmoid and use sigmoid prime
    activation = train_in
    activations = []  # list to store all the activations, layer by layer
    print("train in activations", activations)
    zs = []  # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        print("sigmoid, active", activation)
        activations.append(activation)
        print("weigths", weights)
        print("Final A", activations)

    print(".....................")
    print("Activations: ", activations)

    #print("activation trans ", activations[-2].transpose())

    #   backward pass at Last Layer:
    #   error in output layer:
    #BP1: δL=  ∇aC   ⊙   σ′(zL) output layer error
    deltaL = nabla_cost(activations[-1], train_out) * sigmoid_prime(zs[-1])

    # BP3 ∂C/∂blj = δlj. get the gradient/rate of change in relation to the bias
    dCdb[-1] = deltaL
    dCdw[-1] = np.dot(deltaL, activations[-2].transpose())

    print("Delta cost function: ", deltaL)

    # propogate the error backwards to the other layer
    # BP2 δl=((wl+1)Tδl+1) ⊙ σ′(zl)
    for layer in range(2, len(sizes)):
        #start at 2 to ignore input layer,
        #NOTE: -layer = input layer connections, layer = hidden layer to out connections
        z = zs[-layer] #step backwards toward inputs
        sigp_z = sigmoid_prime(z)

        activations_at_inputs = np.asarray(activations[-layer-1])
        print(" // ", activations_at_inputs)
        print(" /// transposed ", activations_at_inputs.transpose())
        #delta is sent backwards
        delta = np.dot(weights[-layer + 1].transpose(), deltaL) * sigp_z
        dCdb[-layer] = delta
        dCdw[-layer] = np.dot(delta, activations_at_inputs.transpose())
    return(dCdb, dCdw)



def nabla_cost(a, y):
    return a-y



def train(net, trainingdata, gain):
    for i in range(numberofEpochs):
        # net = [inputlayer, layer1, output, connections, biases]
        #learn(biases, weights, data, gain)
        learn(net[4], net[3], trainingdata, gain)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


#RUN


net = initiate()
train(net, trainingdata, 3.0)
#feedforward()
