import math
import numpy as np
#activation functions
def linear(input):
    return input
def sigmoid(input):
    return 1/((1+math.e**(-input)))
def tanh(input):
    return ((math.e**input)-(math.e**(-input)))/((math.e**input) + (math.e**(-input)))
def relu(input):
    return np.maximum(input,np.zeros(input.shape[0]))
def leaky_relu(input,constant=0.01):
    k = np.where(input<=0,input*constant, input)
    return k
def ELU(input,constant=0.01):
    k = np.where(input <= 0, ((math.e ** input) - 1) * constant, input)
    return k
def SELU(input,constant1=1.0507, constant2 = 1.6733):
    return constant1*ELU(input,constant2)
def softmax(input):
    v = np.sum((np.exp(input)), axis=1)
    return np.divide((np.exp(input)), v)
def swish(input):
    return input * sigmoid(input)
def Softplus(input):
    return np.log(1+(math.e**input))
def Mish(input):
    return input * tanh(Softplus(input))

def lineard(input):
    k = input.copy()
    k[:] = 1
    return k
def sigmoidd(input):
    return sigmoid(input) * (1-sigmoid(input))
def tanhd(input):
    return 1- ((tanh(input))**2)
def relud(input):
    k = input.copy()
    k[k>0]=1
    k[k<=0]=0
    return k
def leaky_relud(input,constant=0.01):
    k = np.where(input <= 0, constant, 1)
    return k
def ELUd(input,constant=0.01):
    k = np.where(input <= 0, (math.e ** input) * constant, 1)
    return k
def SELUd(input,constant1=1.0507, constant2 = 1.6733):
    return constant1*ELUd(input,constant2)
def softmaxd(input):
    k = np.empty((input.shape[1],input.shape[1])) #Jacobian Matrix
    sf = softmax(input)
    #print(sf)
    for i in range(input.shape[1]):
        for j in range(input.shape[1]):
            if i == j:
                k[i][j] = sf[0][i]*(1-sf[0][j])
            else:
                k[i][j] = -sf[0][i]*(sf[0][j])
    return k
def swishd(input):
    return sigmoid(input) + (input *(sigmoid(input) * (1 - sigmoid(input))))
def Softplusd(input):
    return sigmoid(input)
def Mishd(input):
    return tanh(Softplus(input)) + ((input*(1-tanh(Softplus(input))**2)) * sigmoid(input))

if __name__ == "__main__":
    z = np.array([[1, 2, 3]])
    print(softmaxd(z))