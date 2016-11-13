from numpy import array, random, dot, exp

"""
This program is a basic 2 layer neural network
input -> input layer -> hidden layer -> output
"""

# LEARNING CODE
def learn(X,y):
    """
    Training phase of a neural network
    Accepts
    X - input
    y - output
    """
    # initialized 2 layers with random weights from -1 to 1
    layer1_weights = 2 * random.random((X.shape[1], 16)) - 1
    layer2_weights = 2 * random.random((16, 1)) - 1
    # train the network for 100000 iterations
    for i in xrange(100000):
        # Find the results of the existing network with current weights and input - Sigmoid function
        layer1_estimation = 1/(1+exp(-(dot(X,layer1_weights))))
        layer2_estimation = 1/(1+exp(-(dot(layer1_estimation,layer2_weights))))
        # Calculate the error - derivative of the Sigmoid function
        layer2_estimation_delta = (y - layer2_estimation)*(layer2_estimation*(1-layer2_estimation))
        layer1_estimation_delta = layer2_estimation_delta.dot(layer2_weights.T) * (layer1_estimation * (1-layer1_estimation))
        # Correct the weights for next pass
        layer2_weights += layer1_estimation.T.dot(layer2_estimation_delta)
        layer1_weights += X.T.dot(layer1_estimation_delta)
    return (layer1_weights, layer2_weights)  # return a tuple for all the weights associated for both layers

# PREDICTION CODE

def predict(X, weights):
    """
    Runs the value in X with a 2 layer neural network with the supplied weights
    """
    layer1_estimation = 1/(1+exp(-(dot(X, weights[0]))))
    layer2_estimation = 1/(1+exp(-(dot(layer1_estimation, weights[1]))))
    return layer2_estimation


# DATA
# Binary AND operator
X1 = array([[0, 0], [0, 1], [1, 0], [1, 1]])
y1 = array([[0, 0, 0, 1]]).T  # T to get transpose of 4x1 matrix

# use the learn method to train over the input and output
and_weights = learn(X1, y1)

test_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
print "Predictions from AND gate trained neural network"
# for each item in the test_set use the trained neural network model to predict
for test_item in test_set:
    and_prediction = predict(test_item, and_weights)
    print str(test_item)+"\t"+str(and_prediction)