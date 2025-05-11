#This is an array of all weight matrices per layer connection - q
import numpy as np
class Layered_Bias:
    def __init__(self, number_of_layer_connections:int,list_of_neurons, output_size, weight_initialization_technique):
        self.list_of_nodes = np.array([])
        for i in range(number_of_layer_connections): #number of hidden layers+1
            if i != number_of_layer_connections-1:
                self.list_of_nodes=np.append(self.list_of_nodes,Biases(weight_initialization_technique, list_of_neurons[i]))
            else:
                self.list_of_nodes=np.append(self.list_of_nodes,Biases(weight_initialization_technique, output_size))

    def getItem(self, item):
        return self.list_of_nodes[item]
    def getList(self):
        return self.list_of_nodes

    def setOne(self, key, value):
        self.list_of_nodes[key].setit(value)

    def setOneSp(self,key,value):
        self.list_of_nodes[key[0]].setOne([(key[1],key[2]),value])
    def append(self, param):
        self.list_of_nodes=np.append(self.list_of_nodes,param)
#1d array for each layer connection. 1 by u
class Biases:
    def __init__(self, weight_initialization_technique, layer_size):
        self.layer_size=layer_size
        if type(layer_size) == tuple:
            layer_size = layer_size[0]
        if weight_initialization_technique == "zero":
            self.bias = np.zeros(1,layer_size)
        elif weight_initialization_technique == "small":
            self.bias = np.full((1,layer_size),0.1)
        elif weight_initialization_technique == "random":
            self.bias = np.random.normal(size=(1,layer_size))
        elif weight_initialization_technique == "uniform":
            self.bias = np.random.uniform(size=(1,layer_size))
        else:
            self.bias = np.random.normal(size=(1, layer_size))

    def getSize(self):
        return self.layer_size
    def setit(self,value):
        self.bias = value
    def setOne(self,key,value):
        self.bias[1][key[0]] = value

    def getL(self):
        return self.bias