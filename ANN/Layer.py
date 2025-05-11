import numpy as np
class Layer:
    #list_of_nodes is numpy array
    def __init__(self, list_of_nodes=None):
        if list_of_nodes is None:
            list_of_nodes = np.array([])
        self.list_of_nodes = np.array(list_of_nodes).reshape(1,len(list_of_nodes))

    def get_l(self):
        return self.list_of_nodes

    #list_of_weights is 2d numpy array, each row being weights corresponding to first node is layer
    #l_nodes -> 1 by n nodes, list_of_weights is n by y, where y is size of hidden layer
    #b is a 1 by y vector
    def linearly_combine(self, l_nodes, list_of_weights,b):
        self.list_of_nodes = np.matmul(l_nodes,list_of_weights) + b

    def apply_activation(self, activation):
        self.list_of_nodes = activation(self.list_of_nodes)

    def set_One(self, key, value):
        self.list_of_nodes[key] = value

    def set_it(self,s):
        self.list_of_nodes = s