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
        weights_input_hidden = np.random.uniform(size = (sizes[0], sizes[1]), low= -1, high=1)
        weights_hidden_output = np.random.uniform(size = (sizes[1], sizes[2]), low= -1, high=1)

        bias_hidden_nodes = np.random.uniform(size = sizes[1], low= -1, high=1)
        bias_output_nodes = np.random.uniform(size = sizes[2], low= -1, high=1)

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
    print("inputs", input)
    print("weights", weights_input_hidden)
    new_hidden_a = np.dot(input, weights_input_hidden) + bias_hidden_nodes
    activation_hidden = sigmoid(new_hidden_a)
    #list.append(activation_hidden)
    print("activations", activation_hidden)
    return activation_hidden

def feedforward_hidden_to_out(previous_layer_activations, weights_hidden_output, bias_output_nodes):
    new_output_a = np.dot(previous_layer_activations, weights_hidden_output) + bias_output_nodes
    activation_out = sigmoid(new_output_a)
    print("activations next layer", activation_out)
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
        for input, output in zip(trainingInput, trainingOutput):
            print("each input: ", input)
            print("each output: ", output)
            #get input in to each node
                #feed forward and get activations
            ai = feedforward_input_to_hidden_layer(input, w_i2h, b_i2h)
            print("activations from input to hidden: ", ai)
            aL = feedforward_hidden_to_out(ai, w_h2o, b_h2o)
            #print("activations last layer: ", aL)
            #print("_______________________________")

            # print("activations into last", ai)
            gradientError = MeasureErrorLastLayer(ai, aL, output, w_h2o, b_h2o)

            deltas = BackPropogate(ai, aL, gradientError, input, w_i2h, b_i2h)
            print("error at hidden layer", deltas[0])

            updatelast = GradientDescent(deltas, w_i2h, b_i2h, w_h2o, b_h2o, len(trainingInput), lr)
            #updatehidden = GradientDescent(HiddenLayerError, w_i2h, b_i2h, len(trainingInput), lr)

            #set the adjusted
            w_h2o = updatelast[0]
            #b_h2o = updatelast[1]

            #w_i2h = updatehidden[0]
            #b_i2h = updatelast[1]

            print("updated: ", updatelast)
            print("_______________________________")

        return net



def MeasureErrorLastLayer(a_h, aL, y, h_weights, l_bias):
    # make a vector of errors from a whole layer, for each neuron j in the layer l
    # error is ej = cost/delta z, delta z a slight perturbation versions of the weighted inputs (z) coming into the neuron j
    # get output activation from hidden layer (aL) by feeding forward again
    a_h = np.asarray(a_h)
    h_weights = np.asarray(h_weights)
    # z is the weighted input the last node (output node)
    z = np.dot(a_h, h_weights) + l_bias
    aL_check = sigmoid(z)
    #confusion here: use activation at last layer or use z into last layer, i think use aL, z used to backprop
    #print("check activation into Last Layer", aL_check, aL, z)

    Cost = y-aL

    gradient = np.dot(Cost, sigmoid_prime(z)) ## here could multiply by learning rate?
    print("a_h", a_h)
    print("h_weights", h_weights)
    print("z", z)
    print("cost gradient", gradient)
    print("----------------end error measure --------------------------")

    return gradient


def BackPropogate(aIn, aL, gradient, inputs, i_weights, h_bias):
    # get the cost gradient from the measure
    # LastLayer Back Prop: this is essentially the first back prop pass
    # Transpose of last layer activations
    al_T = aL.transpose()

    # get deltas to adjust by
    dCdw = al_T * gradient  # * learning rate here? in the future this might need to np.dot if there are more output nodes
    dCdb = gradient
    print("dcdw", dCdw)

    #For last layer:
    print("iweight", i_weights)
    #get z of the inputs into the hidden layer
    z_into_hidden = np.dot(inputs, i_weights) + h_bias # this has to be an np.dot because we sum 6 weights into 3 weighted inputs

    #transpose the input to hidden weight matrix, dot product with errors
    i_weights_T = i_weights.transpose()
    print("iw_t", i_weights_T)

    # to make the np.dot matrix work out ** I think ** I can just do this as a 2d array, but then it might double up as it sums?
    #get next deltas going in to hidden layer:
    dCdw = [dCdw, dCdw]
    print("dcdw", dCdw)

    delta_i = np.dot(i_weights_T, dCdw) * sigmoid_prime(z_into_hidden)
    print("deltas_i", delta_i)

    #activations tranpose
    deltaW_i = np.dot(delta_i, aIn.transpose())
    deltaB_i = delta_i

    layerErrors = [dCdw, dCdb, deltaW_i, deltaB_i]
    return layerErrors

def GradientDescent(layerErrors, w_i2o, b_i2o, w_h2o, b_h2o, batch_size, lr):
    #update the weights by the deltas
    # new_weights = weights + deltas[0] * learningRate
    # new_bias = bias + deltas[1] * learningRate

    print("layer errors++++++++++++++")
    print(layerErrors)
    dCdw = layerErrors[0]
    print("dcdw", dCdw)
    dCdb = layerErrors[1]
    delta_h_w = layerErrors[2]
    print("delta_h_w", delta_h_w)
    delta_h_b = layerErrors[3]
    print("+++++++++++++++++++++++++")


    new_weights = []
    new_bias = []

    for x in range(batch_size):         #might need to randomize/batch the training data sampling here
        # update weights output layer
        for d_weight, w in zip(dCdw, w_h2o):
            new_weight = delta_h_w + w #summation
            nw = w-(lr/batch_size) * new_weight #average out
            #w_h2o = nw
        # update weights at hidden layer
        for w in w_i2o: # this should look like a 6 by 3
            for dw in delta_h_b:
                nw = w+dw
            new_weights.append(nw)
            print("NEW", new_weights)


       # for b_delta, b in zip(delta_L_b, biases):
       #     nb = b-(lr/batch_size) * b_delta
       # new_bias.append(nb)
        update = [new_weights, new_bias]
    return update





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
print("net: ", net)
print("Trained net: ", trained_net)