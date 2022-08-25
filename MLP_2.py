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



        #second verison/matrix hard coded
        weights_input_hidden = [[np.random.uniform(0, 1, 1) for col in range(sizes[1])] for row in range(sizes[0])]
        weights_hidden_output = [[np.random.uniform(0, 1, 1) for col in range(sizes[2])] for row in range(sizes[1])]
        print("wih", weights_input_hidden)
        print("who", weights_hidden_output)

        bias_hidden_nodes = [[np.random.uniform(0, 1, 1) for col in range(sizes[1])] for row in range(1)]
        bias_output_nodes = [[np.random.uniform(0, 1, 1) for col in range(sizes[2])] for row in range(1)]

        print("bih", bias_hidden_nodes)
        print("bho", bias_output_nodes)
         #           0       1        2       3                     4                 5                      6
        net = [inputlayer, layer1, output, weights_input_hidden, bias_hidden_nodes, weights_hidden_output, bias_output_nodes]
        #print(net[2])
        #visualize(net, net[3])
        return net


def feedforward_input_to_hidden_layer(input, weights_input_hidden, bias_hidden_nodes):
    new_hidden_a = np.dot(weights_input_hidden, input) + bias_hidden_nodes
    activation_hidden = sigmoid(new_hidden_a)
    return activation_hidden

def feedforward_hidden_to_out(input, weights_hidden_output, bias_output_nodes):
    new_output_a = np.dot(weights_hidden_output, input) + bias_output_nodes
    activation_out = sigmoid(new_output_a)
    return activation_out

net = initiate()
w_i2h = net[3]
b_i2h = net[4]
w_h2o = net[5]
b_h2o = net[6]

activation_into_hidden = feedforward_input_to_hidden_layer(w_i2h, b_i2h)
activation_into_last = feedforward_hidden_to_out(w_h2o, b_h2o)


def train(trainingOutput):
    for example in trainingOutput:
        LastLayerError = MeasureErrorLastLayer(activation_into_last, example, w_h2o, b_h2o, activation_into_hidden)
    BackPropogateForHiddenError(activation_into_hidden, w_i2h, w_h2o, b_i2h, LastLayerError)
    GradientDescent()
    update()


def MeasureErrorLastLayer(aL, y, z, h_weights, l_bias, aIn):
    #difference between one component of target out and one component of activation
    #derivative of quadratic cost function is a simple difference
    CostL = y-aL
    #TODO: make sure that y target lines up correct with activation outs/guesses
    # this should not be a for loop but a parrallel iterate (matrix difference)

    #get z (weighted input) same as feedforward without sigmoid
    #can use feedforward for Z or does it have to remove sigmoid forward?
    z = np.dot(h_weights, aL) + l_bias
    deltaL = np.dot(CostL, sigmoid_prime(z))

    #get gradient of out to hidden weight and bias for last Layer
    dCdw_L = aIn * deltaL
    dCdb_L = deltaL

    lastlayerError = [dCdw_L, dCdb_L]
    return lastlayerError


def BackPropogateForHiddenError(aIn, i_weights, h_weights, h_bias, lastLayerError):
    #pass in the deltaL from the output layer error calc
    deltaL = lastLayerError[1]
    print("passed in delta L: ", deltaL)

    #get z of the inputs into the hidden layer
    z_hidden = np.dot(i_weights, aIn) + h_bias

    #transpose the hidden to output weight matrix, dot product with errors
    #TODO: double check that I correctly used transpose of the weights on the layer forward
    # to get error backward
    # and check that z of the current (hidden) layer is correctly used below:
    transpose1 = h_weights.transpose()
    print("weights ", h_weights)
    print("transpose1 ", transpose1)

    #get hidden layer errors:
    delta_h = np.dot(transpose1, deltaL) * sigmoid_prime(z_hidden)
    dCdb_h = aIn * delta_h
    dCdw_h = delta_h

    hiddenlayerError = [dCdw_h, dCdb_h]
    return hiddenlayerError

def GradientDescent():
    

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

