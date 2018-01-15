import numpy as np
from copy import deepcopy

def integerToBinary(num):
    return np.unpackbits(np.arange(num, num+1, dtype='>i%d' %(1)).view(np.uint8))[-4:]

def getTrainData():
    X = np.array([[1] + list(integerToBinary(i)) for i in range(0,16)]).T
    D = np.array([bin(i).count("1")%2!=0 for i in range(0,16)]).reshape(1,16).astype(int)
    return X,D

def intializeWeights():
    return [np.random.uniform(-1, 1, (5, 4))] + [np.random.uniform(-1, 1, (5, 1))]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def forwardProp(x,W_layers):
    output = []
    output.append(sigmoid(np.dot(W_layers[0].T,x)))
    output.append(sigmoid(np.dot(W_layers[1].T,np.insert(output[0],0,1,axis=0))))
    return np.array(output)

def backProp(x,Y,d_k,W,eta,delta_w,alpha):
    y_k = Y[1]

    delta_k = y_k*(1-y_k)*(d_k-y_k)
    delta_w[1] = eta*delta_k.T*np.insert(Y[0],0,1,axis=0)+(alpha*delta_w[1])

    delta_j = Y[0]*(1-Y[0])*W[1][1:,:]*delta_k
    delta_w[0] = eta*delta_j.T*x+(alpha*delta_w[0])

    W[0] += delta_w[0]
    W[1] += delta_w[1]


def mlp(X,D,W,eta,alpha):
    epoch = 0
    delta_w = [np.zeros((5, 4))] + [np.zeros((5, 1))]

    while True:
        isError = False
        for i in range(0,16):
            x = X[:,i].reshape(5,1)
            Y = forwardProp(x,W)
            backProp(x, Y, np.reshape(D[:, i], (1, 1)), W, eta, delta_w, alpha)

            if np.abs(np.squeeze(D[:,i]-Y[1])) > 0.05:
                isError = True
        epoch += 1
        if not isError:
            break

    return epoch

if __name__ == "__main__":

    X,D = getTrainData()
    W = intializeWeights()
    for eta in np.arange(0.05, 0.51, 0.05):
        print("eta : ", eta, ", momentum : ", 0, " epoch : ", mlp(X, D, deepcopy(W), eta, 0))
        print("eta : ", eta, ", momentum :", 0.9, " epoch : ", mlp(X, D, deepcopy(W), eta, 0.9))