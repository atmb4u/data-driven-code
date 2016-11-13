from numpy import array, random, dot, exp

inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = array([[0, 1, 1, 0]]).T

def learn(X,y):
    layer1_weights = 2 * random.random((X.shape[1], 16)) - 1
    layer2_weights = 2 * random.random((16, 1)) - 1
    for j in xrange(10000):
        layer1_estimation = 1/(1+exp(-(dot(X,layer1_weights))))
        layer2_estimation = 1/(1+exp(-(dot(layer1_estimation,layer2_weights))))
        layer2_estimation_delta = (y - layer2_estimation)*(layer2_estimation*(1-layer2_estimation))
        layer1_estimation_delta = layer2_estimation_delta.dot(layer2_weights.T) * (layer1_estimation * (1-layer1_estimation))
        layer2_weights += layer1_estimation.T.dot(layer2_estimation_delta)
        layer1_weights += X.T.dot(layer1_estimation_delta)
    return (layer1_weights, layer2_weights)

xor_weights = learn(inputs, outputs)

def predict(X, weights):
    layer1_estimation = 1/(1+exp(-(dot(X, weights[0]))))
    layer2_estimation = 1/(1+exp(-(dot(layer1_estimation, weights[1]))))
    return layer2_estimation

test_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
for test_item in test_set:
    xor_prediction = predict(test_item, xor_weights)
    print str(test_item)+"\t"+str(xor_prediction)