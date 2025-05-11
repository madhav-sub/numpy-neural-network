import math
import numpy as np
def SGD(old,lr,derivative):
    return old - (lr*derivative)
def MomenSGD(beta,old,lr,derivative,vold):
    return old - (lr*((beta*vold) + ((1-beta) * derivative))), ((beta*vold) + ((1-beta) * derivative))

def Adam(t, beta1,beta2,old,lr,derivative,mold,vold):
    m = beta1*mold + (1-beta1)*derivative
    v = beta2*vold + (1-beta2)*(derivative**2)
    mhat = m/((1-(beta1 ** t))+1e-9)
    vhat = v/((1-(beta2 ** t))+1e-9)
    return old - ((lr*mhat)/(np.sqrt(vhat) + 1e-9)), m, v

def Adagrad(Gold, old, lr, derivative):
    G = Gold + (derivative)**2
    return old-((lr/(np.sqrt(G+1e-9)))*derivative), G
def RMSprop(beta,Gold, old, lr, derivative):
    G = (beta*Gold) + (1-beta)*((derivative)**2)
    return old-((lr/(np.sqrt(G)+1e-9))*derivative), G