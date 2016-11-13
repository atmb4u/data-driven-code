from numpy import array, random, dot, exp

"""
This program is a basic 2 layer neural network
input -> input layer -> hidden layer -> output

"""

# DATA
# Binary AND operator
X1 = array([[0, 0], [0, 1], [1, 0], [1, 1]])
y1 = array([[0, 0, 0, 1]]).T

# LEARNING CODE
def learn(X,y):
    """
    Training phase of a neural network
    Accepts an 
    X - input, 
    y - output
    """
    # initialized 2 layers with random weights from -1 to 1
    layer1_weights = 2 * random.random((X.shape[1], 16)) - 1
    layer2_weights = 2 * random.random((16, 1)) - 1
    # train the network for 100000 iterations
    for i in xrange(100000):
        # Find the results of the existing network with current weights and input - Sigmoid function
        l1 = 1/(1+exp(-(dot(X,layer1_weights))))
        l2 = 1/(1+exp(-(dot(l1,layer2_weights))))
        # Calculate the error - derivative of the Sigmoid function
        l2_delta = (y - l2)*(l2*(1-l2))
        l1_delta = l2_delta.dot(layer2_weights.T) * (l1 * (1-l1))
        # Correct the weights for next pass
        layer2_weights += l1.T.dot(l2_delta)
        layer1_weights += X.T.dot(l1_delta)
    return (layer1_weights, layer2_weights)  # return a tuple for all the weights associated for both layers

# PREDICTION CODE

def predict(X, weights):
    l1 = 1/(1+exp(-(dot(X, weights[0]))))
    l2 = 1/(1+exp(-(dot(l1, weights[1]))))
    return l2


and_weights = learn(X1, y1)
test_set = [[0, 0], [0, 1], [1, 0], [1, 1]]

for test_item in test_set:
    and_prediction = predict(test_item, and_weights)
    print str(test_item)+"\t"+str(and_prediction)