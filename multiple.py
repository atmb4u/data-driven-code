try:
    import numpy as np
except ImportError:
    print "Install numpy - pip install numpy"
"""
This program is a basic 2 layer neural network
input -> input layer -> hidden layer -> output

There are 4 different examples, simulating AND, OR, XOR and NAND gates with the same code.
"""

# DATA
# Binary AND operator
X1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y1 = np.array([[0, 0, 0, 1]]).T
# Binary OR operator
X2= np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y2 = np.array([[0, 1, 1, 1]]).T
# Binary XOR operator
X3= np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y3 = np.array([[0, 1, 1, 0]]).T
# Binary NAND operator
X4= np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y4 = np.array([[1, 1, 1, 0]]).T


# LEARNING CODE

def learn(X,y,n,h=16):
	"""
	Training phase of a neural network
	Accepts an 
	X - input, 
	y - output
	n - number of training iterations, 		
	h - number of hidden neurons
	"""
	# initialized 2 layers with random weights from -1 to 1
	layer1_weights = 2 * np.random.random((X.shape[1], h)) - 1
	layer2_weights = 2 * np.random.random((h, 1)) - 1
	# train the network for n iterations
	for i in xrange(n):
		# Find the results of the existing network with current weights and input - Sigmoid function
	    l1 = 1/(1+np.exp(-(np.dot(X,layer1_weights))))
	    l2 = 1/(1+np.exp(-(np.dot(l1,layer2_weights))))
	    # Calculate the error - derivative of the Sigmoid function
	    l2_delta = (y - l2)*(l2*(1-l2))
	    l1_delta = l2_delta.dot(layer2_weights.T) * (l1 * (1-l1))
	    # Correct the weights for next pass
	    layer2_weights += l1.T.dot(l2_delta)
	    layer1_weights += X.T.dot(l1_delta)
	return (layer1_weights, layer2_weights)  # return a tuple for all the weights associated for both layers

# PREDICTION CODE

def predict(X, weights):
	l1 = 1/(1+np.exp(-(np.dot(X, weights[0]))))
	l2 = 1/(1+np.exp(-(np.dot(l1, weights[1]))))
	return l2

# Same neural network code used to learn about different gates


and_weights = learn(X1, y1, 10000)
# print and_weights
or_weights = learn(X2,y2, 10000)
# print or_weights
xor_weights = learn(X3,y3, 10000)
# print xor_weights
nand_weights = learn(X4,y4, 10000)
# print xor_weights
test_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
print "ITEM\tAND\tOR\tXOR\tNAND"
for test_item in test_set:
	and_prediction = predict(test_item, and_weights)
	or_prediction = predict(test_item, or_weights)
	xor_prediction = predict(test_item, xor_weights)
	nand_prediction = predict(test_item, nand_weights)
	# round will round off the floting point results to the nearest number (in this case, 0 or 1)
	print str(test_item)+"\t"+str(round(and_prediction))+"\t"+str(round(or_prediction))+ \
		"\t"+str(round(xor_prediction))+"\t"+str(round(nand_prediction))