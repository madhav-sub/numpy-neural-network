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
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets


print(X.columns)
print(X.shape)
print(y.shape)
print(X.dtypes)
print(y.dtypes)
print(y.value_counts())

lr=[]
nb=[]
ann=[]
from sklearn.model_selection import train_test_split
for seed in range(10):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)

    y_train2 = np.array(pd.get_dummies(y_train).astype(int)).reshape((y_train.shape[0],3))
    y_test2 = np.array(pd.get_dummies(y_test).astype(int)).reshape((y_test.shape[0],3))
    print(y_train.columns)

    X_train,X_test,y_train,y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
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

    y_train=y_train2.copy()
    del y_train2
    y_test=y_test2.copy()
    del y_test2

    model = ANN(False, False, "random", "random", len(list(X.columns)),
                [(16, leaky_relu),(16, leaky_relu),(16, leaky_relu),(16, leaky_relu)], 3, softmax,
                training_args={"optimizers": Adam, "loss": CCE, "lr": 0.00025, "epochs": 35})
    line_x = list(range(3981 * 35))
    plt.scatter(x=line_x, y=model.train(X_train, y_train), s=4)

    plt.show()

    c = 0
    for i in range(X_test.shape[0]):
        if np.argmax(model.test(X_test[i])) == np.argmax(y_test[i]):
                c += 1
    ann.append(c / X_test.shape[0])

    print(seed, lr, nb,ann)


lr = [0.49887133182844245, 0.49887133182844245, 0.49887133182844245, 0.49887133182844245, 0.49887133182844245, 0.49887133182844245, 0.49887133182844245, 0.49887133182844245,0.49887133182844245, 0.49887133182844245]
nb = [0.7133182844243793, 0.6681715575620768, 0.672686230248307, 0.6749435665914221, 0.6749435665914221, 0.690744920993228, 0.6772009029345373, 0.6839729119638827, 0.6704288939051919, 0.6772009029345373]
ann = [0.46952595936794583, 0.6930022573363431, 0.5237020316027088, 0.6952595936794582, 0.7268623024830699, 0.7404063205417607, 0.5665914221218962, 0.7200902934537246,0.618510158013544, 0.7133182844243793]
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