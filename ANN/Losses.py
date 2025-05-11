import numpy as np
#loss functions
def MSE(pred, label):
    return np.mean(np.square(label-pred))

def MAE(pred, label):
    return np.mean(np.abs(label-pred))

def LCL(pred, label):
    return np.mean(np.log(np.cosh(label-pred)))

def BCE(pred, label):
    return np.mean(-label*np.log(pred+1e-9) - (1-label)*np.log((1-pred)+1e-9))

def CCE(pred, label):
    return -np.mean(np.sum(label * np.log(pred+1e-9)))

def MSEd(pred, label):
    return -2*np.mean(label-pred)

def MAEd(pred, label):
    return -np.mean(np.sign(label-pred))

def LCLd(pred, label):
    return -np.mean(np.tanh(label-pred))

def BCEd(pred, label):
    return np.mean((pred-label)/((pred*(1-pred))+1e-9),axis=0)

def CCEd(pred, label):
    return -np.mean(label/(pred+1e-9),axis=0)


