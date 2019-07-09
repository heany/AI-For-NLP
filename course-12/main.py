# build nueral networks from sratch

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import  shuffle, resample


class Node:
    def __init__(self, inputs=[]):
        self.inputs = inputs
        self.outputs = []

        for n in self.inputs:
            n.outputs.append(self)
            # set 'self' node as input_node's output_node

        self.value = None
        self.gradients = {}
        # keys are the inputs to this node, and their
        # values are the partials of this node with
        # respect to that input.

    def forward(self):
        '''
        Forward propagation.
        Compute the output value vased on 'input_nodes' and
        store the result in self.value
        :return:
        '''

        raise NotImplemented

    def backward(self):

        raise NotImplemented

class Input(Node):
    def __init__(self):
        '''
        An Input node has no input_nodes.
        So no need to pass anything to the Node instantiator.
        '''

        Node.__init__(self)

    def forward(self, value = None):
        '''
        Only input node is the node where the value may be passed
        as an argument to forward().
        All other node implementations should get the value of the
        previous node from self.input_nodes
        example:
        val0: self.inputs_node[0].value
        :return:
        '''
        if value is not None:
            self.value = value
            # It's is input node, when need to forward, this node initiate self's value.

            # Input subclass just holds a value,
            # such as a data feature or a model parameter(weight/bias)

    def backward(self):
        self.gradients = {self:0}
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost * 1

        # input N --> N1,N2
        # \partial L / \Partial N
        # ==> \partial L / \partial N1 * \ partial N1 / \partial N

class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)

    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inputs))
        # when execute forward, this node calculate value as defined.

class Linear(Node):
    def __init__(self, nodes, weights, bias):
        Node.__init__(self, [nodes, weights, bias])

    def forward(self):
        inputs = self.inputs[0].value
        weights = self.inputs[1].value
        bias = self.inputs[2].value

        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        # initial a partial for each of the input nodes
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            # get the partial of the cost w.r.t this node
            grad_cost = n.gradients[self]
            self.gradients[self.inputs[0]] = np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] = np.dot(self.inputs[0].value.T, grad_cost)
            self.gradients[self.inputs[2]] = np.sum(grad_cost, axis = 0, keepdims=False)

            # WX + B / W ==> X
            # WX + B / X ==> W

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / ( 1 + np.exp(-1 * x))

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self._sigmoid(self.x)

    def backward(self):
        self.partial = self._sigmoid(self.x) * ( 1- self._sigmoid(self.x))

        # y = 1 / ( 1 + e^-x)
        # y' = 1 / ( 1 + e^-x) * (1 - 1 / (1 + e^-x))

        self.gradients = {n:np.zeros_like(n.value) for n in self.inputs}
        for n in self.outputs:
            grad_cost = n.gradients[self] # get the partial of the cost with respect to this node.
            self.gradients[self.inputs[0]] = grad_cost * self.partial
            # use * to keep all the dimension same

class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inputs[0].value.reshape(-1,1)
        a = self.inputs[1].value.reshape(-1,1)
        assert(y.shape == a.shape)

        self.m = self.inputs[0].value.shape[0]
        self.diff = y - a

        self.value = np.mean(self.diff**2)

    def backward(self):
        self.gradients[self.inputs[0]] = (2 / self.m) * self.diff
        self.gradients[self.inputs[1]] = (-2 / self.m) * self.diff

def forward_and_backward(outputnode, graph):
    # excute all the forward method of sorted_nodes.
    # In partice, it's common to feed in multiple data example in each forward pass rather than just 1.
    # Because the examples can be processed in paralle1. the number of examples is called batch size

    for n in graph:
        n.forward()
        # each node execute forward, get self.value based on the topological sort result

    for n in graph[::-1]:
        n.backward()

    # return outputnode.value

# v --> a --> C
# b --> C
# b --> v -- a --> C
# v --> v --> a --> C

def topological_sort(feed_dict):
    '''
    Sort generic nodes in topological order using Kahn's Algorithm.
    :param feed_dict: A dictionary where the key is a 'Input' node
                    and the value is the respective value feed to that node.
    :return: A list of sorted nodes.
    '''

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out':set()}
        for m in n.outputs:
            if m not in G:
                G[m] = {'in':set(), 'out':set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
            # if n is Input Node, set n'value as feed_dict[n]
            # else, n's value is calculate as its input_nodes

        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def sgd_update(trainables, learning_rate = 1e-2):
    # there are so many other update / optimization methods
    # such as Adam, Mom,
    for t in trainables:
        t.value += -1 * learning_rate * t.gradients[t]

# =========================== start to train ========
'''
Check out the new network architecture and dataset
Notice that the weights and biases are generated randomly
No need to change anything, but feel free to tweak to test your network, 
play around with the epochs, batch size, etc. 
'''

# load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 5000
# total number of example
m = X_.shape[0]
batch_size = 16
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainables = [W1,b1,W2,b2]

losses = []

print("The size of example = {}".format(m))
print(graph)
print(trainables)

# step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # step 1
        # randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # step 2
        _ = None
        forward_and_backward(_, graph) # set output node not important.

        # step 3
        rate = 1e-2
        sgd_update(trainables, rate)

        loss += graph[-1].value
    if i % 100 == 0:
        print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
        losses.append(loss)

import matplotlib.pyplot as plt
plt.plot(range(len(losses)),losses)

