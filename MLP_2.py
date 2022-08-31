import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

sizes = [2,3,1]

initA = 0
initB = 1

numberofEpochs = 5000
lr = 1.0 #learningn rate

trainingInput = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
trainingOutput = [[0.0], [1.0], [1.0], [0.0]]

trainingdata = zip(trainingInput, trainingOutput)
trainingdata_manual = [[0.0, 0.0], [0.0],
                       [0.0, 1.0], [1.0],
                       [1.0, 0.0], [1.0],
                       [1.0, 1.0], [0.0]]


def initiate():
    for inputlayer in range(sizes[0]):
        inputlayer = np.array([initA, initB])
        #for layer1 in range(sizes[1]):
        #    layer1 = np.random.uniform(-1, 1, sizes[1])
        #for output in range(sizes[2]):
        #    output = np.random.uniform(-1, 1, sizes[2])

    weights_input_hidden = np.random.uniform(size = (sizes[0], sizes[1]), low= -1, high=1)
    weights_hidden_output = np.random.uniform(size = (sizes[1], sizes[2]), low= -1, high=1)

    bias_hidden_nodes = np.random.uniform(size = sizes[1], low= -1, high=1)
    bias_output_nodes = np.random.uniform(size = sizes[2], low= -1, high=1)

    print("wi2h", weights_input_hidden)
    print("wh2o", weights_hidden_output)
    print("bi2h", bias_hidden_nodes)
    print("bh2o", bias_output_nodes)

         #           0               1               2                   3                    4
    net = [inputlayer, weights_input_hidden, bias_hidden_nodes, weights_hidden_output, bias_output_nodes]
    print("inital net", net)
        #visualize(net, net[3])
    return net


def feedforward_input_to_hidden_layer(input, weights_input_hidden, bias_hidden_nodes):
    new_hidden_a = np.dot(input, weights_input_hidden) + bias_hidden_nodes
    activation_hidden = sigmoid(new_hidden_a)
    return activation_hidden

def feedforward_hidden_to_out(previous_layer_activations, weights_hidden_output, bias_output_nodes):
    new_output_a = np.dot(previous_layer_activations, weights_hidden_output) + bias_output_nodes
    activation_out = sigmoid(new_output_a)
    return activation_out



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
    w_i2h = net[1]
    b_i2h = net[2]
    w_h2o = net[3]
    b_h2o = net[4]

    for e in range(numberofEpochs):
        print("**************Running Epoch****** ", e)
        for input, output in zip(trainingInput, trainingOutput):
            print("current input: ", input)
            print("current output: ", output)
            #get input in to each node
            #feed forward and get activations
            ah = feedforward_input_to_hidden_layer(input, w_i2h, b_i2h)
            print("activations from input to hidden: ", ah)
            aL = feedforward_hidden_to_out(ah, w_h2o, b_h2o)
            print("activations from hidden to out: ", aL)

            #BP(aIn, aL, error_L, inputs, i2h_weights, h2o_weights, i2h_bias, h2o_bias):
            deltas = BackPropogate(ah, aL, input, output, w_i2h, b_i2h, w_h2o, b_h2o)

            update = GradientDescent(deltas, w_i2h, b_i2h, w_h2o, b_h2o, len(trainingInput), lr)
            print("net", net)
            newNet = updateNet(update, net)
            print("trained net", net)
            print("_______________________________")
    return newNet



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


def BackPropogate(ah, aL, inputs, y, w_i2h, b_i2h, w_h2o, b_h2o):
    # get the cost gradient from the measure
    # LastLayer Back Prop: this is essentially the first back prop pass
    # Transpose of last layer weights
    w_into_o_T = w_h2o.transpose()
    w_into_h_T = w_i2h.transpose()

    #get zs
    z_into_hidden = np.dot(inputs, w_i2h) + b_i2h # this has to be an np.dot because we sum 6 weights into 3 weighted inputs
    z_into_out = np.dot(ah, w_h2o) + b_h2o

    #activation at output node
    #aL = sigmoid(z_into_out) # same as what is already passed in from feedforward, ao
    #aH = sigmoid(z_into_hidden)
    #print("aL_Check", aL_check, "al", aL)
    #print("Ain Check", aH_check, "ah", ah)
    Cost = y-aL

    gradient_L = np.dot(Cost, aL) ## here could multiply by learning rate?

    #get deltas for last layer
    dcdw_o = aL * gradient_L
    dcdb_o = gradient_L

    #back prop to middle layer
    # δl = ((wl + 1)T * δ l+1) ⊙ σ′(zl),
    #despite above formula, using weights into hidden makes more sense to me here?
    #print("transpose of the weight", w_into_o_T)
    #print("transpose of next weights", w_into_h_T)
    error_h = np.dot(w_into_o_T, gradient_L) * sigmoid_prime(z_into_hidden)

    # get deltas to adjust next layer by
    dcdw_h = ah.transpose() * error_h  # * tranpose of aIN? correct aIns?
    dcdb_h = error_h

    layerErrors = [dcdw_h, dcdb_h, dcdw_o, dcdb_o]
    print("____current weight Errors_____" )
    print("weight errors at hidden: ", dcdw_h)
    print("bias errors at hidden: ", dcdb_h)
    print("weight errors at hidden: ", dcdw_o)
    print("bias errors at hidden: ", dcdb_o)
    return layerErrors

def GradientDescent(layerErrors, w_i2h, b_i2h, w_h2o, b_h2o, batch_size, lr):
    #update the weights by the deltas
    # new_weights = weights + deltas[0] * learningRate
    # new_bias = bias + deltas[1] * learningRate

    print("// d, deltas // errors //")
    dw_h = layerErrors[0]
    db_h = layerErrors[1]
    dw_o = layerErrors[2]
    db_o = layerErrors[3]
    print("-------")

    new_weights_o = []
    new_weights_h = []
    new_bias_o = []
    new_bias_h = []

    ##-----just tests------##
    #w_i2h_t = w_i2h.transpose()
    #print("w_i2h", w_i2h)
    #print("w_i2h_tranposed", w_i2h_t)
    #print("dw_h", dw_h)
    #for ws in w_i2h:
    #    for dws in dw_h:
    #        for w, dw in zip(ws, dws):
    #           print("w, dw", w, dw)
    ##---------------------------##

    for x in range(1):  # might need to randomize/batch the training data sampling here
        # update weights hidden layer
        for ws in w_i2h:
            for dws in dw_h:
                for w, dw in zip(ws, dws):
                    #print("w, dw", w, dw)
                    summed_l1 = dw + w  # summation
                    nw = w - (lr / batch_size) * summed_l1 # average out
                    #print("each weight", w,  "delta weight", dw, "new weight", nw)
                    new_weights_h.append(nw)
        print("new_weights into hidden", new_weights_h)
        # update weights at output layer
        #there is only one output delta for all 3 weights (hardcoded here)
        for ws in w_h2o:
            for w in ws:
                for dw in dw_o: #print("each w_h2o", w)
                    summed_l2 = w + dw
                    nw_1 = w - (lr / batch_size) * summed_l2 # average out
                    new_weights_o.append(nw_1)
        print("new weights into output", new_weights_o)

        # update bias at hidden layer
        #print("b_i2h", b_i2h, "db_h", db_h)
        for dbs in db_h:
            for b, db in zip(b_i2h, dbs):
                #print("each b_i2h", b, "each db_h", db)
                summedb_l2 = b + db
                b_1 = b - (lr / batch_size) * summedb_l2  # average out
                new_bias_h.append(b_1)
        print("new bias into hidden nodes", new_bias_h)

        #update bias at output
        #print("b_h2o", b_h2o, "db_o", db_o)
        b = b_h2o
        summedb_l1 = db_o + b  # summation
        nbs = b - (lr / batch_size) * summedb_l1 # average out
        for nb in nbs:
            new_bias_o.append(nb)
        print("new bias at output node", new_bias_o)


    update = [new_weights_h, new_bias_h, new_weights_o, new_bias_o]

    return update

def updateNet(update, net):

    print("...updating net")
    net[1] = update[0]
    net[2] = update[1]
    net[3] = update[2]
    net[4] = update[3]
    return net

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

net = initiate()
#w_i2h = net[1]  #6
#b_i2h = net[2]  #3
#w_h2o = net[3]  #3
#b_h2o = net[4]  #1

print("Intialized Net: ", net)
print("------------------------------")

trained_net = train(net, trainingInput, trainingOutput)
#print("net: ", net)
#print("Trained net: ", trained_net)