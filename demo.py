from numpy import array, random, dot, exp

inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = array([[0], [1], [1], [0]])

def learn(X, y):
    l1_w = 2 * random.random((X.shape[1], 16)) - 1
    l2_w = 2 * random.random((16, 1)) - 1
    for j in xrange(10000):
        l1 = 1 / (1 + exp(-(dot(X, l1_w))))
        l2 = 1 / (1 + exp(-(dot(l1, l2_w))))
        l2_delta = (y - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(l2_w.T) * (l1 * (1-l1))
        l2_w += l1.T.dot(l2_delta)
        l1_w += X.T.dot(l1_delta)
    return (l1_w, l2_w)

xor_weights = learn(inputs, outputs)

def predict(X, weights):
    l1 = 1/(1+exp(-(dot(X, weights[0]))))
    l2 = 1/(1+exp(-(dot(l1, weights[1]))))
    return l2

test_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
for test_item in test_set:
    xor_prediction = predict(test_item, xor_weights)
    print str(test_item)+"\t"+str(xor_prediction)