import itertools
import os

from sklearn.linear_model import LogisticRegression

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sympy.physics.units import newton
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
color_cycle = itertools.cycle(colors)

from Final_Project.ANN.Activations import sigmoid, tanh, relu, leaky_relu, ELU, SELU, softmax, swish, \
    Softplus, Mish, sigmoidd, tanhd, relud, leaky_relud, ELUd, SELUd, softmaxd, swishd, Softplusd, Mishd, \
    linear, lineard
from Final_Project.ANN.Losses import MSE, MAE, LCL, BCE, CCE, MSEd, MAEd, LCLd, BCEd, CCEd
from Final_Project.ANN.Layer import Layer
from Final_Project.ANN.Layered_Weights import Layered_Weights
from Final_Project.ANN.Layered_Bias import Layered_Bias
from Final_Project.ANN.Optimizers import SGD, MomenSGD, Adam, Adagrad, RMSprop
import numpy as np
import matplotlib.pyplot as plt

TF_ENABLE_ONEDNN_OPTS=0
#optimizers
dict = {sigmoid: sigmoidd, tanh: tanhd, relu: relud, leaky_relu: leaky_relud, ELU: ELUd, SELU: SELUd, softmax: softmaxd,
        swish: swishd, Softplus: Softplusd, Mish: Mishd, linear: lineard}
dict2 = {MSE: MSEd, MAE: MAEd, LCL: LCLd, BCE: BCEd, CCE: CCEd}


#Random
class ANN:
    def __init__(self, is_generator, is_discriminator, weight_initialization_technique_weights, weight_initialization_technique_bias, input_size,
                 list_of_neurons: list, output_size, output_activation, training_args=None):
        self.new_div = []
        self.labels = None
        if training_args is None:
            training_args = {"optimizer": SGD, "loss": MSE, "lr": 0.01, "epochs": 3}  #Default training args
        if True:
            self.is_generator = is_generator
            self.is_discriminator = is_discriminator
            self.losses = []
            self.input_layer = None  #default input layer is just None until we add input layer. When initialized, 1 by input size array.
            self.input_layer_size = input_size  #Input layer size
            self.hidden_layers = np.array([])  #Creates structure that contains the hidden layers (Size num_hid_layers)
            for i in range(len(list_of_neurons)):
                self.hidden_layers = np.append(self.hidden_layers, Layer(
                    [0] * list_of_neurons[i][0]))  #Generate Hidden Layers for each set of connections. Initially None
            self.hidden_layers_sizes = [k[0] for k in
                                        list_of_neurons]  #List of sizes of each hidden layer (Size num_hid_layers)
            self.output_layer = Layer()  #Creates output_layer. When initialized, 1 by input size array.
            self.output_layer_size = output_size  #Size of output layer
            self.special_list_activations = [k[1] for k in list_of_neurons] + [
                output_activation]  #all activation functions per layer connections (Size num_hid_layers + 1)
            self.training_args = training_args  #All training args set by user
            self.weights = Layered_Weights(len(self.hidden_layers_sizes) + 1, self.input_layer_size,
                                           self.hidden_layers_sizes, self.output_layer_size,
                                           weight_initialization_technique_weights)  #Creates Weights for each connection of layers. There are num_hid_layers + 1 connections, and so we have a numpy array of 2d arrays, with the 2d arrays being the number of incoming nodes and columns being the number of outgoing nodes in a connections.
            self.intermediate_derivatives = Layered_Weights(len(self.hidden_layers_sizes) + 1, self.input_layer_size,
                                                            self.hidden_layers_sizes, self.output_layer_size,
                                                            None)  #Creates Weights for each connection of layers. There are num_hid_layers + 1 connections, and so we have a numpy array of 2d arrays, with the 2d arrays being the number of incoming nodes and columns being the number of outgoing nodes in a connections.
            self.intermediate_derivatives2 = Layered_Weights(len(self.hidden_layers_sizes) + 1, self.input_layer_size,
                                                             self.hidden_layers_sizes, self.output_layer_size,
                                                             None)  #Creates Weights for each connection of layers. There are num_hid_layers + 1 connections, and so we have a numpy array of 2d arrays, with the 2d arrays being the number of incoming nodes and columns being the number of outgoing nodes in a connections.
            self.biases = Layered_Bias(len(self.hidden_layers_sizes) + 1, self.hidden_layers_sizes,
                                       self.output_layer_size,
                                       weight_initialization_technique_bias)  #Bias is applied at the layers themselves, being all hidden layers and the output, and so it is of size (,num_hid_layers + 1). Each bias is a 1 by number of nodes in current layer matrix.
            #We finally have derivatives for each node wrt the previous node for instance. The reason for this is due to the ease of calculating the chain role
            self.derivatives = Layered_Weights(len(self.hidden_layers_sizes) + 1, self.input_layer_size,
                                               self.hidden_layers_sizes, self.output_layer_size,
                                               None)  #Creates Weights for each connection of layers. There are num_hid_layers + 1 connections, and so we have a numpy array of 2d arrays, with the 2d arrays being the number of incoming nodes and columns being the number of outgoing nodes in a connections.

    def train(self, inputs, labels):  #must be a numpy array of shape (input_layer_size,), similar for labels
        self.losses = []
        m=[]
        v=[]
        g=[]
        m1 = []
        v1 = []
        g1 = []
        if inputs.shape[1] != self.input_layer_size or labels.shape[1] != self.output_layer_size:
            print("Error! Sizes do not match!")
            return None
        for _ in range(self.training_args['epochs']):
            #print("epoch:", _+1)
            with tqdm(total=inputs.shape[0], desc=f"Epoch {_ + 1} Progress",ncols = 100) as pbar:
                for inputl in range(inputs.shape[0]):
                    self.input_layer = Layer(list_of_nodes=inputs[inputl])
                    self.labels = Layer(list_of_nodes=labels[inputl])
                    #Has derivative of L wrt each node, for each hidden layer and for output layer
                    self.forward_propagation() #sets derivatives, output layer, hidden layers based on current weights
                    m,v,g,m1,v1,g1 = self.backpropagate(m,v,g,m1,v1,g1,inputl)
                    pbar.update(1)
                    if self.is_discriminator and labels[0][0] == 0:
                        self.new_div.append(np.array(dict2[self.training_args['loss']](self.output_layer.get_l(), np.array([[1]]))).reshape(self.output_layer_size, 1))
                        for f in range(len(self.hidden_layers_sizes)+1):  #Need to look through all hidden layers, not input layers
                            self.new_div.append(np.matmul(self.derivatives.getItem(len(self.hidden_layers_sizes) - f).getL(), self.new_div[-1]))
                            #self.new_div[f] = np.maximum(np.minimum(self.new_div[f],1),-1) #Gradient Clipped
                        valuecric = self.new_div[-1]
                        vvaluecric1 = self.training_args['loss'](self.output_layer.get_l(), np.array([[1]]))
                        self.new_div = []
                        return valuecric, vvaluecric1
                color = next(color_cycle)
                pbar.colour = color
        print(self.losses[0])
        return self.losses

    def forward_propagation(self):  #Input can be a 1D list of inputs, only one array for elements
        for s, j in enumerate(self.hidden_layers):
            if s == 0:
                j.linearly_combine(self.input_layer.get_l(), self.weights.getItem(s).getL(),self.biases.getItem(s).getL())  #sets the values of the first hidden layer
                if self.is_discriminator:
                    self.derivatives.setOne(0, self.weights.getItem(0).getL() * np.tile(dict[self.special_list_activations[0]](j.get_l()), (self.input_layer_size,1)))  #finds derivative of next output wrt inputs. We have inp by out * inp by out, as 1 by out was converted to repeat imp times for each weight
                self.intermediate_derivatives.setOne(s, np.matmul(self.input_layer.get_l().T,dict[self.special_list_activations[0]](j.get_l())))  #inp by 1 * 1 by out, do/dw
                self.intermediate_derivatives2.setOne(s, dict[self.special_list_activations[0]](j.get_l()))
                j.apply_activation(self.special_list_activations[s])
            else:
                j.linearly_combine(self.hidden_layers[s - 1].get_l(), self.weights.getItem(s).getL(),self.biases.getItem(s).getL())
                self.derivatives.setOne(s, self.weights.getItem(s).getL() * np.tile(dict[self.special_list_activations[s]](j.get_l()), (self.hidden_layers_sizes[s - 1],1)))  #finds derivative of next output wrt inputs. We have inp by out * inp by out, as 1 by out was converted to repeat imp times for each weight
                self.intermediate_derivatives.setOne(s, np.matmul(self.hidden_layers[s - 1].get_l().T,dict[self.special_list_activations[s]](j.get_l())))
                self.intermediate_derivatives2.setOne(s, dict[self.special_list_activations[s]](j.get_l()))  #inp by 1 of 1s * 1 by out, do/db
                j.apply_activation(self.special_list_activations[s])
        self.output_layer.linearly_combine(self.hidden_layers[-1].get_l(), self.weights.getItem(-1).getL(),self.biases.getItem(-1).getL())
        if self.special_list_activations[-1] != softmax:
            self.derivatives.setOne(-1, self.weights.getItem(-1).getL() * np.tile(dict[self.special_list_activations[-1]](self.output_layer.get_l()), (self.hidden_layers_sizes[-1],1)))  #finds derivative of next output wrt inputs. We have inp by out * inp by out, as 1 by out was converted to repeat imp times for each weight
            self.intermediate_derivatives.setOne(-1, np.matmul(self.hidden_layers[-1].get_l().T,dict[self.special_list_activations[-1]](self.output_layer.get_l())))
            self.intermediate_derivatives2.setOne(-1, dict[self.special_list_activations[-1]](self.output_layer.get_l()))  #inp by 1 of 1s * 1 by out, do/db
        else:
            self.derivatives.setOne(-1, np.matmul(self.weights.getItem(-1).getL(),softmaxd(self.output_layer.get_l()).T))
            zero_matrix = np.zeros_like(softmaxd(self.output_layer.get_l()))
            np.fill_diagonal(zero_matrix, np.diag(softmaxd(self.output_layer.get_l())))
            self.intermediate_derivatives.setOne(-1, np.matmul(np.tile(self.hidden_layers[-1].get_l(),(self.output_layer_size,1)).T,zero_matrix))
            self.intermediate_derivatives2.setOne(-1, np.diag(softmaxd(self.output_layer.get_l())).reshape(1,self.output_layer_size))  # inp by 1 of 1s * 1 by out, do/db
        self.output_layer.apply_activation(self.special_list_activations[-1])
        if not self.is_generator:
            #if(self.training_args['loss'](self.output_layer.get_l(), self.labels.get_l())>1):
            #print("LOSS1:", self.training_args['loss'](self.output_layer.get_l(), self.labels.get_l()))
            self.new_div.append(np.array(dict2[self.training_args['loss']](self.output_layer.get_l(), self.labels.get_l())).reshape(self.output_layer_size,1))
            self.losses.append(self.training_args['loss'](self.output_layer.get_l(), self.labels.get_l()))
        if self.is_generator:
            return self.output_layer.get_l()

    def backpropagate(self,m,v,g,m1,v1,g1,inputl=0,loss_layer=None):
        if loss_layer is not None: #Only for Generator
            print(self.new_div)
            self.new_div.append(loss_layer[0])
            self.losses.append(loss_layer[1])
            print("LOSS2:", loss_layer[1])
        for f in range(len(self.hidden_layers_sizes)):  # Need to look through all hidden layers, not input layers
            self.new_div.append(np.matmul(self.derivatives.getItem(len(self.hidden_layers_sizes) - f).getL(), self.new_div[-1]))
            #self.new_div[f] = np.maximum(np.minimum(self.new_div[f], 1), -1)  # Gradient Clipped
        self.new_div = self.new_div[::-1]
        for k in range(len(self.weights.getList())):
            if self.training_args['optimizers'] == SGD:
                d = self.intermediate_derivatives.getItem(k).getL() * np.tile(self.new_div[k].T, (
                self.intermediate_derivatives.getItem(k).getL().shape[0], 1))
                self.weights.setOne(k, SGD(self.weights.getItem(k).getL(), self.training_args['lr'], d))
            elif self.training_args['optimizers'] == MomenSGD:
                if len(v) < len(self.weights.getList()):
                    v.append(np.zeros_like(self.weights.getItem(k).getL()))
                d = self.intermediate_derivatives.getItem(k).getL() * np.tile(self.new_div[k].T, (
                self.intermediate_derivatives.getItem(k).getL().shape[0], 1))
                yout1, v[k] = MomenSGD(0.9, self.weights.getItem(k).getL(), self.training_args['lr'], d, v[k])
                self.weights.setOne(k, yout1)
            elif self.training_args['optimizers'] == Adam:
                if len(v) < len(self.weights.getList()):
                    v.append(np.zeros_like(self.weights.getItem(k).getL()))
                if len(m) < len(self.weights.getList()):
                    m.append(np.zeros_like(self.weights.getItem(k).getL()))
                d = self.intermediate_derivatives.getItem(k).getL() * np.tile(self.new_div[k].T, (
                self.intermediate_derivatives.getItem(k).getL().shape[0], 1))
                yout1, m[k], v[k] = Adam(inputl + 1, 0.9, 0.999, self.weights.getItem(k).getL(),
                                         self.training_args['lr'], d, m[k], v[k])
                self.weights.setOne(k, yout1)
            elif self.training_args['optimizers'] == Adagrad:
                if len(g) < len(self.weights.getList()):
                    g.append(np.zeros_like(self.weights.getItem(k).getL()))
                d = self.intermediate_derivatives.getItem(k).getL() * np.tile(self.new_div[k].T, (
                self.intermediate_derivatives.getItem(k).getL().shape[0], 1))
                yout1, g[k] = Adagrad(g[k], self.weights.getItem(k).getL(), self.training_args['lr'], d)
                self.weights.setOne(k, yout1)
            elif self.training_args['optimizers'] == RMSprop:
                if len(g) < len(self.weights.getList()):
                    g.append(np.zeros_like(self.weights.getItem(k).getL()))
                d = self.intermediate_derivatives.getItem(k).getL() * np.tile(self.new_div[k].T, (
                self.intermediate_derivatives.getItem(k).getL().shape[0], 1))
                yout1, g[k] = RMSprop(0.9, g[k], self.weights.getItem(k).getL(), self.training_args['lr'], d)
                self.weights.setOne(k, yout1)
        for l in range(len(self.biases.getList())):
            if self.training_args['optimizers'] == SGD:
                d = self.intermediate_derivatives2.getItem(l).getL() * self.new_div[l].T
                self.biases.setOne(l, SGD(self.biases.getItem(l).getL(), self.training_args['lr'], d))
            elif self.training_args['optimizers'] == MomenSGD:
                if len(v1) < len(self.biases.getList()):
                    v1.append(np.zeros_like(self.biases.getItem(l).getL()))
                d = self.intermediate_derivatives2.getItem(l).getL() * np.tile(self.new_div[l].T, (
                self.intermediate_derivatives2.getItem(l).getL().shape[0], 1))
                yout1, v1[l] = MomenSGD(0.9, self.biases.getItem(l).getL(), self.training_args['lr'], d, v1[l])
                self.biases.setOne(l, yout1)
            elif self.training_args['optimizers'] == Adam:
                if len(v1) < len(self.biases.getList()):
                    v1.append(np.zeros_like(self.biases.getItem(l).getL()))
                if len(m1) < len(self.biases.getList()):
                    m1.append(np.zeros_like(self.biases.getItem(l).getL()))
                d = self.intermediate_derivatives2.getItem(l).getL() * np.tile(self.new_div[l].T, (
                self.intermediate_derivatives2.getItem(l).getL().shape[0], 1))
                yout1, m1[l], v1[l] = Adam(inputl + 1, 0.9, 0.999, self.biases.getItem(l).getL(),
                                           self.training_args['lr'], d, m1[l], v1[l])
                self.biases.setOne(l, yout1)
            elif self.training_args['optimizers'] == Adagrad:
                if len(g1) < len(self.biases.getList()):
                    g1.append(np.zeros_like(self.biases.getItem(l).getL()))
                d = self.intermediate_derivatives2.getItem(l).getL() * np.tile(self.new_div[l].T, (
                self.intermediate_derivatives2.getItem(l).getL().shape[0], 1))
                yout1, g1[l] = Adagrad(g1[l], self.biases.getItem(l).getL(), self.training_args['lr'], d)
                self.biases.setOne(l, yout1)
            elif self.training_args['optimizers'] == RMSprop:
                if len(g1) < len(self.biases.getList()):
                    g1.append(np.zeros_like(self.biases.getItem(l).getL()))
                d = self.intermediate_derivatives2.getItem(l).getL() * np.tile(self.new_div[l].T, (
                self.intermediate_derivatives2.getItem(l).getL().shape[0], 1))
                yout1, g1[l] = RMSprop(0.9, g1[l], self.biases.getItem(l).getL(), self.training_args['lr'], d)
                self.biases.setOne(l, yout1)
        self.new_div = []
        return m,v,g,m1,v1,g1

    def test(self, inputs):  # Input can be a 1D list of inputs, only one array for elements
        if inputs.shape[0] != self.input_layer_size:
            print("Error! Sizes do not match!")
            return None
        self.input_layer = Layer(list_of_nodes=inputs)
        for s, j in enumerate(self.hidden_layers):
            if s == 0:
                j.linearly_combine(self.input_layer.get_l(), self.weights.getItem(s).getL(),
                                   self.biases.getItem(s).getL())  # sets the values of the first hidden layer
                j.apply_activation(self.special_list_activations[s])
            else:
                j.linearly_combine(self.hidden_layers[s - 1].get_l(), self.weights.getItem(s).getL(),
                                   self.biases.getItem(s).getL())
                j.apply_activation(self.special_list_activations[s])
        self.output_layer.linearly_combine(self.hidden_layers[-1].get_l(), self.weights.getItem(-1).getL(),
                                           self.biases.getItem(-1).getL())
        self.output_layer.apply_activation(self.special_list_activations[-1])
        return self.output_layer.get_l()


from tensorflow import keras
from keras.datasets import mnist

if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    ti = train_images.reshape(train_images.shape[0], -1)
    tl = train_labels.reshape(train_images.shape[0], 1)

    tei = test_images.reshape(test_images.shape[0], -1)
    tel = test_labels.reshape(test_labels.shape[0], 1)

    from sklearn.metrics import accuracy_score

    #Gassuain Naive Bayes
    m = GaussianNB()
    m.fit(ti, tl.ravel())
    print("Gaussian NB")
    print(accuracy_score(tel.ravel(), m.predict(tei)))

    clf = LogisticRegression(
        solver='saga',
        max_iter=10,
        verbose=1
    )
    print("Logistic Regression")
    clf.fit(ti, tl.ravel())
    print(accuracy_score(tel.ravel(), clf.predict(tei)))

    tl = keras.utils.to_categorical(train_labels, num_classes=10)
    tel = keras.utils.to_categorical(test_labels, num_classes=10)

    model = ANN(False, False,"random", "random", 784, [(128, leaky_relu),(128, leaky_relu),(128, leaky_relu)], 10, softmax,
                training_args={"optimizers": Adam, "loss": CCE, "lr": 0.0001, "epochs": 3})
    line_x = list(range(60000*3))

    print("ANN")

    plt.scatter(x=line_x,y=model.train(ti, tl),s=4)
    c = 0
    for i in range(10000):
       if np.argmax(model.test(tei[i])) == np.argmax(tel[i]):
          c+=1
    print(c/10000)
    plt.show()


