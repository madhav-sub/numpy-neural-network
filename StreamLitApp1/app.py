import numpy as np
import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

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

    m = GaussianNB()
    m.fit(X_train, y_train.ravel())
    print("Gaussian NB")

    print("Logistic Regression")
    m.fit(X_train, y_train.ravel())
    print(accuracy_score(y_test.ravel(), m.predict(X_test)))

f1,f2,f3,f4,f5,f6,f7= st.number_input("Perimeter", min_value=0.0,max_value=200.0),st.number_input("Major Axis Length", min_value=0.0,max_value=200.0),st.number_input("Minor Axis Length", min_value=0.0,max_value=200.0),st.number_input("Eccentricity", min_value=0.0,max_value=200.0),st.number_input("Extent", min_value=0.0,max_value=200.0),st.number_input("Convex Area", min_value=0,max_value=200),st.number_input("Area", min_value=0,max_value=200)

features = [f7,f1,f2,f3,f4,f5,f6]
if st.button("Predict"):
    print(m.predict(np.array([features])))
    st.write("Prediction: ", str(m.predict(np.array([features]))).replace("['","").replace("']",""))