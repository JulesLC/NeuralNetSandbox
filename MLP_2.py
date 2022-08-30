import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

sizes = [2,3,1]

initA = 0
initB = 1

numberofEpochs = 1
lr = 3.0 #learningn rate

trainingInput = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
trainingOutput = [[0.0], [1.0], [1.0], [0.0]]

trainingdata = zip(trainingInput, trainingOutput)
trainingdata_manual = [[0.0, 0.0], [0.0],
                       [0.0, 1.0], [1.0],
                       [1.0, 0.0], [1.0],
                       [1.0, 1.0], [0.0]]


def initiate():
     for layer in sizes:
        for inputlayer in range(sizes[0]):
            inputlayer = np.array([initA, initB])
        for layer1 in range(sizes[1]):
            layer1 = np.random.uniform(-1, 1, sizes[1])
        for output in range(sizes[2]):
            output = np.random.uniform(-1, 1, sizes[2])

        #second verison/matrix hard coded
        weights_input_hidden = [np.random.uniform(size = (sizes[0], sizes[1]), low= -1, high=1)]
        weights_hidden_output = [np.random.uniform(size = (sizes[1], sizes[2]), low= -1, high=1)]

        bias_hidden_nodes = [np.random.uniform(size = sizes[1], low= -1, high=1)]
        bias_output_nodes = [np.random.uniform(size = sizes[2], low= -1, high=1)]

        print("wih", weights_input_hidden)
        print("who", weights_hidden_output)
        print("bih", bias_hidden_nodes)
        print("bho", bias_output_nodes)

         #           0       1        2       3                     4                 5                      6
        net = [inputlayer, layer1, output, weights_input_hidden, bias_hidden_nodes, weights_hidden_output, bias_output_nodes]
        #print(net[2])
        #visualize(net, net[3])
        return net


def feedforward_input_to_hidden_layer(input, weights_input_hidden, bias_hidden_nodes):
    #list = []
    new_hidden_a = np.dot(input, weights_input_hidden) + bias_hidden_nodes
    activation_hidden = sigmoid(new_hidden_a)
    #list.append(activation_hidden)
    return activation_hidden

def feedforward_hidden_to_out(previous_layer_activations, weights_hidden_output, bias_output_nodes):
    new_output_a = np.dot(previous_layer_activations, weights_hidden_output) + bias_output_nodes
    activation_out = sigmoid(new_output_a)
    return activation_out


net = initiate()
w_i2h = net[3]
b_i2h = net[4]
w_h2o = net[5]
b_h2o = net[6]

print("Intialized Net: ", net)
print("------------------------------")

def test(data, weights):
    newtest = []
    for x in range(2):
        for each in data:
            #print("each", each)
            for i in each:
                #print("i", i)
                for w in weights:
                    #print("W", w)
                    test = np.dot(i, w)+2
                newtest.append(test)
        return newtest


def train(net, trainingInput, trainingOutput):
    w_i2h = net[3]
    b_i2h = net[4]
    w_h2o = net[5]
    b_h2o = net[6]

    for e in range(numberofEpochs):
        for input in trainingInput:
            #get input in to each node
            print("each input: ", input)
            #feed forward and get activations
            ai = feedforward_input_to_hidden_layer(input, w_i2h, b_i2h)
            print("activations from input to hidden: ", ai)
            aL = feedforward_hidden_to_out(ai, w_h2o, b_h2o)
            print("activations last layer: ", aL)
        print("_______________________________")

        for output in trainingOutput:
            print("each output: ", output)
            print("activations being measured", ai)
            LastLayerError = MeasureErrorLastLayer(ai, output, w_h2o, b_h2o)
            print("error at last layer", LastLayerError)
            deltas = BackPropogate(ai, w_i2h, w_h2o, b_i2h, LastLayerError)
            deltas_w = deltas[0]
            deltas_b = deltas[1]
            GradientDescent_weights(3.0, deltas_w, len(trainingInput), w_h2o)
            GradientDescent_weights(3.0, deltas_w, len(trainingInput), w_i2h)
            GradientDescent_bias(3.0, deltas_b, len(trainingInput), b_h2o)
            GradientDescent_bias(3.0, deltas_b, len(trainingInput), b_i2h)
        return net


def MeasureErrorLastLayer(input, y, h_weights, l_bias):
    # make a vector of errors from a whole layer, for each neuron j in the layer l
    # error is ej = cost/delta z, delta z a slight perturbation versions of the weighted inputs (z) coming into the neuron j

    # get output activation from hidden layer (aL) by feeding forward again
    z = np.dot(input, h_weights) + l_bias
    aL = sigmoid(z) #is activation supposed to be the one passed through the sigmoid, or is it supposed to be z without sigmoid (to use sigmoid prime)?

    Cost = y-aL

    gradient = np.dot(Cost, sigmoid_prime(z)) ## here multiply by learning rate?

    #get deltas to w, b adjust by
    delta_L_w = aL * gradient # * learning rate here?
    delta_L_b = gradient

    lastlayerdeltas = [delta_L_w, delta_L_b]
    return lastlayerdeltas


def BackPropogate(aIn, i_weights, h_weights, h_bias, lastLayerError):
    # get the gradient from the cost, and move it through

    #outputs (check) aIn = aL.tranpose()?
    #error of this layer = transpose of weight matrix from last layer(l+1) * error from last layer, haddamard multipy with sigmoid prime z of this layer (l)
    #pass in the deltaL from the output layer error calc
    deltaW = lastLayerError[1]
    print("passed in delta L: ", deltaW)

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
    delta_h = np.dot(transpose1, deltaW) * sigmoid_prime(z_hidden)
    dCdb_h = aIn * delta_h
    dCdw_h = delta_h

    hiddenlayerError = [dCdw_h, dCdb_h]
    return hiddenlayerError

def GradientDescent_weights(learningRate, deltas, size_total_training, weights):
    #update the weights by the deltas

    # new_weights = weights + deltas[0] * learningRate
    # new_bias = bias + deltas[1] * learningRate

    new_weights = []
    updated_weights = []
    for x in range(size_total_training):
        #add the gradients to the weights
        for w_deltas,w in zip(deltas[0], weights):
            nw = w_deltas + w
            new_weights.append(nw)
            for n in new_weights:
                next_weight = w-(learningRate/size_total_training)*n
                updated_weights.append(next_weight)
            return updated_weights


#Left off here:
def GradientDescent_bias(learningRate, deltas, size_total_training, bias):
    #update the weights by the deltas
    new_biases = []
    updated_biases = []
    for x in range(size_total_training):
        #add the gradients to the weights
        for b_deltas,b in zip(deltas[1], bias):
            bw = b_deltas + b
            new_biases.append(bw)
            for n in new_biases:
                next_weight = b-(learningRate/size_total_training)*n
                updated_biases.append(next_weight)
            return updated_biases


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def visualize(nodes, edges):
    # visualization
    G = nx.Graph()
    #for e in edges:
      #  G.add_edge_value(e.split(","))
    for n in nodes:
        G.add_node(n)
    nx.draw(G, with_labels=True)
    plt.show()




xdata = [[1, 1], [0, 0]]
xweights = [[5, 5, 5], [5, 5, 5]]
print("----- test")
print("   test", test(xdata, xweights))
print("________________________________")

trained_net = train(net, trainingInput, trainingOutput)
print("Trained net: ", trained_net)