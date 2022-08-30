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

            print("W_h2o", w_h2o)
            #gradientError = MeasureErrorLastLayer(ai, aL, output, w_h2o, b_h2o)
            #BP(aIn, aL, error_L, inputs, i2h_weights, h2o_weights, i2h_bias, h2o_bias):
            deltas = BackPropogate(ai, aL, input, output, w_i2h, w_h2o, b_i2h, b_h2o)
            print("error at hidden layer", deltas[0])

            update = GradientDescent(deltas, w_i2h, b_i2h, w_h2o, b_h2o, len(trainingInput), lr)

            print("updated: ", update)
            #set the adjusted
            #w_h2o = updatelast[0]
            #b_h2o = updatelast[1]

            #w_i2h = updatehidden[0]
            #b_i2h = updatelast[1]


            print("_______________________________")

        return net



def MeasureErrorLastLayer(ai, ao, y, h_weights, l_bias):
    # make a vector of errors from a whole layer, for each neuron j in the layer l
    # error is ej = cost/delta z, delta z a slight perturbation versions of the weighted inputs (z) coming into the neuron j
    # get output activations from hidden layer to output node (aL) by feeding forward again
    ai = np.asarray(ai)
    h_weights = np.asarray(h_weights)
    print("ai", ai, "h_wights", h_weights)
    # z is the weighted input the last node (output node)
    z = np.dot(ai, h_weights) + l_bias
    aL_check = sigmoid(z)
    #confusion here: use activation at last layer or use z into last layer, i think use aL, z used to backprop
    print("check output activation into Last Layer", aL_check, ao, z)

    Cost = y-ao

    error_L= np.dot(Cost, sigmoid_prime(z)) ## here could multiply by learning rate?

    print("cost gradient", error_L)
    print("----------------end error measure --------------------------")

    return error_L


def BackPropogate(aIn, aL, inputs, y, i2h_weights, h2o_weights, i2h_bias, h2o_bias):
    # get the cost gradient from the measure
    # LastLayer Back Prop: this is essentially the first back prop pass
    # Transpose of last layer weights
    w_into_o = w_h2o.transpose()

    #get zs
    z_into_hidden = np.dot(inputs, i2h_weights) + i2h_bias # this has to be an np.dot because we sum 6 weights into 3 weighted inputs
    z_into_out = np.dot(aIn, h2o_weights) + h2o_bias

    #activation at output node
    ao = sigmoid(z_into_out)
    Cost = y-ao

    error_L = np.dot(Cost, sigmoid_prime(z_into_out)) ## here could multiply by learning rate?

    #get deltas for last layer
    dcdw_o = sigmoid(z_into_out) * error_L
    dcdb_o = error_L

    #back prop to middle layer
    # δl = ((wl + 1)T * δ l+1) ⊙ σ′(zl),
    error_hidden = np.dot(w_into_o, error_L) * sigmoid_prime(z_into_hidden)

    # get deltas to adjust next layer by
    dcdw_h = sigmoid(z_into_hidden).transpose() * error_hidden  # * tranpose of aIN? correct aIns?
    dcdb_h = error_hidden

    layerErrors = [dcdw_o, dcdb_o, dcdw_h, dcdb_h]

    # to make the np.dot matrix work out ** I think ** I can just do this as a 2d array, but then it might double up as it sums?
    #get next deltas going in to hidden layer:

    #w_into_H = w_i2h.transpose()
    #delta_i = np.dot(w_into_H, dCdb) * sigmoid_prime(z_into_hidden)

    #activations tranpose
    #deltaW_i = np.dot(delta_i, aIn.transpose())
    #deltaB_i = delta_i
    #print("D delta Weights", deltaW_i)
    #print("D delta Bias", deltaB_i)
    #layerErrors = [dCdw, dCdb, deltaW_i, deltaB_i]
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
    delta_i_w = layerErrors[2]
    print("delta_h_w", delta_i_w)
    delta_i_b = layerErrors[3]
    print("delta_h_b", delta_i_b)
    print("+++++++++++++++++++++++++")

    new_weights_o = []
    new_weights_h = []
    new_bias_o = []
    new_bias_h = []

    #print("www", w_h2o) = 3
    #print("www2", w_i2o) = 6
    for x in range(1):  # might need to randomize/batch the training data sampling here
        # update weights output layer
        for w in w_h2o:
            summed_l1 = dCdw[0] + w  # summation
            nw = w - (lr / batch_size) * summed_l1 # average out
        new_weights_o.append(nw)
        # update weights at hidden layer
        for w in w_i2o:  # this should look like a 6 by 3
            summed_l2 = w + delta_i_w
            nw_1 =  w - (lr / batch_size) * summed_l2 # average out
        new_weights_h.append(nw_1)

        #same for bias
        #update bias at output
        #print("ddddelta bias", dCdb, "bbbb", b_h2o ) # since there is only one I can hard code with no for loop, if there are more outputs this will need to for loop
        b = b_h2o
        summedb_l1 = dCdb + b  # summation
        nb = b - (lr / batch_size) * summedb_l1 # average out
        new_bias_o.append(nb)
        # update bias at hidden layer
        print("delta_hb", delta_i_b)
        for bi in b_i2o:  # this should look like a 6 by 3
            for db in delta_i_b:
                 summedb_l2 = bi + db
                 b_1 =  bi - (lr / batch_size) * summedb_l2 # average out
            new_bias_h.append(b_1)
    print("NEW_weight_to hidden layer", new_bias_h)
    print("NEW weights to output layer", new_weights_o)

    update = [new_weights_h, new_bias_h, new_weights_o, new_bias_o]
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