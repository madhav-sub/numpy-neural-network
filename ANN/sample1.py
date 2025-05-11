import pickle

from tensorflow import keras
from keras.datasets import mnist
import itertools
import os

from sklearn.linear_model import LogisticRegression

from Final_Project.ANN.ANN import ANN

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
lr=[]
nb=[]
ann=[]

for seed in range(1):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    ti = train_images.reshape(train_images.shape[0], -1)
    tl = train_labels.reshape(train_images.shape[0], 1)

    tei = test_images.reshape(test_images.shape[0], -1)
    tel = test_labels.reshape(test_labels.shape[0], 1)


    from sklearn.metrics import accuracy_score
    # Gaussian Naive Bayes
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

    model = ANN(False, False, "random", "random", 784, [(128, leaky_relu), (128, leaky_relu), (128, leaky_relu)], 10,
                softmax,
                training_args={"optimizers": Adam, "loss": CCE, "lr": 0.0001, "epochs": 3})
    line_x = list(range(60000 * 3))

    print("ANN")
    model.train(ti, tl)

    with open('../StreamLitApp1/model1.pkl', 'wb') as f:
        pickle.dump(model, f)
    plt.scatter(x=line_x, y=s, s=4)
    print(seed, lr, nb,ann)

    try:
        c = 0
        for i in range(10000):
            if np.argmax(model.test(tei[i])) == np.argmax(tel[i]):
                c += 1
        ann.append(c / 10000)

        plt.show()
    except:
        ann.append("IGNORE")
        seed-=1
        print("HI")


print(lr)
print(nb)
print(ann)

lr = [0.9256, 0.9261, 0.9252, 0.9253, 0.9258, 0.926, 0.9262, 0.9264, 0.9259, 0.925]
nb = [0.5558, 0.5558, 0.5558, 0.5558, 0.5558, 0.5558, 0.5558, 0.5558, 0.5558, 0.5558]
ann = [0.9546, 0.9545, 0.9534, 0.9479, 0.9521, 0.9534, 0.9492, 0.9563, 0.9575, 0.9584]
from scipy.stats import wilcoxon, friedmanchisquare, f_oneway

if "IGNORE" not in ann:
    print("Logistic vs NB", wilcoxon(lr,nb))
    print("ANN vs NB", wilcoxon(ann,nb))
    print("ANN vs Logistic", wilcoxon(ann,lr))

    s,p=friedmanchisquare(ann,lr,nb)
    print("Friedman ", s, p)

    s, p = f_oneway(ann, lr, nb)
    print("anova p-value:", p)