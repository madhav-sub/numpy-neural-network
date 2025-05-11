import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from ANN import ANN
import itertools
import os
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
from ucimlrepo import fetch_ucirepo

# fetch dataset
rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

# data (as pandas dataframes)
X = rice_cammeo_and_osmancik.data.features
y = rice_cammeo_and_osmancik.data.targets

print(X.shape)
print(y.shape)
print(X.dtypes)
print(y.dtypes)

lr=[]
nb=[]
ann=[]

for seed in range(1):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,stratify=y,random_state=42)

    y_train2 = np.array(pd.get_dummies(y_train).astype(int)).reshape((y_train.shape[0],2))
    y_test2 = np.array(pd.get_dummies(y_test).astype(int)).reshape((y_test.shape[0],2))

    X_train,X_test,y_train,y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
    print(y_train.shape)
    print(X_train.shape)

    # Gassuain Naive Bayes
    m = GaussianNB()
    m.fit(X_train, y_train.ravel())
    print("Gaussian NB")
    print(accuracy_score(y_test.ravel(), m.predict(X_test)))

    clf = LogisticRegression(
        solver='saga',
        max_iter=20,
        verbose=1
    )
    print("Logistic Regression")
    clf.fit(X_train, y_train.ravel())
    print(accuracy_score(y_test.ravel(), clf.predict(X_test)))
    with open('../StreamLitApp1/model1.pkl', 'wb') as f:
        pickle.dump(clf, f)
    y_train=y_train2.copy()
    del y_train2
    y_test=y_test2.copy()
    del y_test2

    model = ANN(False, False, "random", "random", len(list(X.columns)),
                [(64, leaky_relu),(128, leaky_relu),(64, leaky_relu)], 2, softmax,
                training_args={"optimizers": Adam, "loss": CCE, "lr": 0.0001, "epochs": 10})
    line_x = list(range(3429 * 10))
    plt.scatter(x=line_x, y=model.train(X_train, y_train), s=4)
    plt.show()

    c = 0
    for i in range(X_test.shape[0]):
        if np.argmax(model.test(X_test[i])) == np.argmax(y_test[i]):
            c += 1
    ann.append(c / X_test.shape[0])

    print(seed, lr, nb,ann)

print(lr)
print(nb)
print(ann)

from scipy.stats import wilcoxon, friedmanchisquare, f_oneway

print("Logistic vs NB", wilcoxon(lr,nb))
print("ANN vs NB", wilcoxon(ann,nb))
print("ANN vs Logistic", wilcoxon(ann,lr))

s,p=friedmanchisquare(ann,lr,nb)
print("Friedman ", s, p)

s, p = f_oneway(ann, lr, nb)
print("anova p-value:", p)