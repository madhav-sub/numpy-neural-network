import numpy as np
import math
class Layered_Weights:
    def __init__(self, number_of_layer_connections:int,input_size, list_of_neurons, output_size, weight_initialization_technique):
        '''
        :param number_of_layer_connections: number of weight matrices
        :param input_size: first layer size
        :param list_of_neurons: sizes of hidden layers
        :param output_size: output layer size
        :param weight_initialization_technique: weight initializing
        '''
        self.list_of_nodes = np.array([])
        for i in range(number_of_layer_connections):
            if i == 0:
                self.list_of_nodes=np.append(self.list_of_nodes,Weights(weight_initialization_technique, input_size, list_of_neurons[0]),)
            elif i != number_of_layer_connections-1:
                self.list_of_nodes=np.append(self.list_of_nodes,Weights(weight_initialization_technique, list_of_neurons[i-1], list_of_neurons[i]))
            else:
                self.list_of_nodes=np.append(self.list_of_nodes,Weights(weight_initialization_technique, list_of_neurons[i-1], output_size))
    def getItem(self, item):
        return self.list_of_nodes[item]

    def getList(self):
        return self.list_of_nodes

    def setOne(self,key,value):
        self.list_of_nodes[key].setit(value)

    def setOneSp(self,key,value):
        self.list_of_nodes[key[0]].setOne([(key[1],key[2]),value])
    def append(self, param):
        k = Weights(None,param.shape[0],param.shape[1])
        k.setit(param)
        self.list_of_nodes=np.append(self.list_of_nodes, k)


#2d arrays for each layer connection. Each column is for each node in output layer and each row is for input layer   n by u
class Weights:
    def __init__(self, weight_initialization_technique, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        if weight_initialization_technique == "zero":
            self.weights = np.zeros(shape=(in_size,out_size))
        elif weight_initialization_technique == "random normal":
            self.weights = np.random.normal(size=(in_size,out_size))
        elif weight_initialization_technique == "random uniform":
            self.weights = np.random.uniform(size=(in_size,out_size))
        elif weight_initialization_technique == "random poisson":
            self.weights = np.random.poisson(size=(in_size,out_size))
        elif weight_initialization_technique == "random exponential":
            self.weights = np.random.exponential(size=(in_size,out_size))
        elif weight_initialization_technique == "xavier_glorot_uniform":
            self.weights = np.random.uniform(low = -math.sqrt(6/(in_size+out_size)),high = math.sqrt(6/(in_size+out_size)),size=(in_size,out_size))
        elif weight_initialization_technique == "xavier_glorot_norm":
            self.weights = np.random.normal(scale = math.sqrt(6/(in_size+out_size)),size=(in_size,out_size))
        elif weight_initialization_technique == "he_uniform":
            self.weights = np.random.uniform(low = -math.sqrt(6/in_size),high = math.sqrt(6/out_size),size=(in_size,out_size))
        elif weight_initialization_technique == "he_norm":
            self.weights = np.random.normal(scale = math.sqrt(2/in_size),size=(in_size,out_size))
        else:
            self.weights = np.zeros(shape=(in_size,out_size))
    def getInSize(self):
        return self.in_size
    def getOutSize(self):
        return self.out_size
    def setOne(self,key,value):
        self.weights[key[0]][key[1]] = value
    def setit(self,value):
        self.weights = value

    def getL(self):
        return self.weights