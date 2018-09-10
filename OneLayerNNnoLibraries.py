import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of hidden nodes
num_nodes = 100
#number of outputs
num_outputs = 10
model = {}
model['W1'] = np.random.randn(num_nodes,num_inputs) / np.sqrt(num_inputs)
model['C'] = np.random.randn(num_outputs,num_nodes) / np.sqrt(num_nodes)
model['b1'] = np.random.randn(num_nodes,1)/np.sqrt(num_nodes)
model['b2'] = np.random.randn(num_outputs,1)/np.sqrt(num_outputs)
model_grads = copy.deepcopy(model)
model_fin_diff_dir = copy.deepcopy(model)

def sig(z):
    return 1 / (1 + np.exp(-z))
def sigp(z):
    # return 1 / (1+ np.exp(-x))
    # return z*(z>0)
    return sig(z)*(1-sig(z))
def relu(z):
    z[z<0]=0
    return z
    # return z * (z > 0)
    # return np.maximum(z,0)
def relup(z):
    arr1 = np.zeros(num_nodes)
    for i in range(num_nodes):
        if z[i] ==0:
             arr1[i]=0
        else:
            arr1[i] = 1
    return arr1

def genonehot(y):
    arr = np.zeros(num_outputs)
    arr[y] = 1
    return arr

#  Following softmax function from https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
def softmax_function(z):
    return np.exp(z)/np.sum(np.exp(z), axis=0)
def finite_difference(x,y, model):
    de = 0.00001
    # first term
    Z = np.dot(model['W1']+de*model['W1'], x).reshape(num_nodes,1)+model['b1']
    H = relu(Z)
    U = np.dot(model['C'],H).reshape(num_outputs,1)+model['b2']
    f1 = softmax(U)
    # second term
    Z = np.dot(model['W1']-de*model['W1'], x).reshape(num_nodes,1)+model['b1']
    H = relu(Z)
    U = np.dot(model['C'],H).reshape(num_outputs,1)+model['b2']
    f2 = softmax(U)

    return (f1-f2)/(2*delta)

def forward(x,y, model):
    Z = np.dot(model['W1'], x).reshape(num_nodes,1)+model['b1']
    H = relu(Z)
    U = np.dot(model['C'],H).reshape(num_outputs,1)+model['b2']
    f = softmax(U)
    return f,H,Z
def backward(x,y,p, model, model_grads,H,Z):
    # for i in range(len(num_outputs)):
    dpdu = -(genonehot(y).reshape(10,1)-p)
    dpdb2 = dpdu
    dpdc = np.dot(dpdu,H.transpose())
    delta = np.dot(model['C'].transpose(),dpdu)
    dpdb1 = np.multiply(delta,relup(Z).reshape(num_nodes,1))
    dpdw = dpdb1*x.transpose()
    # temp = finite_difference(x,y,model)
    # print("And this is finite diff estimate of dpdw",temp)
    for i in range(num_nodes):
        model_grads['W1'][i,:] = dpdw[i,:]
        model_grads['b1'][i,:] = dpdb1[i,:]
    for i in range(num_outputs):
        model_grads['C'][i,:] = dpdc[i,:]
        model_grads['b2'][i,:] = dpdb2[i,:]
    return model_grads
import time
time1 = time.time()
LR = .05
num_epochs = 20
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.01
    if (epochs > 10):
        LR = 0.005
    if (epochs > 15):
        LR = 0.001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        p,H,Z = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,p, model, model_grads, H,Z)
        model['W1'] = model['W1'] - LR*model_grads['W1']
        model['C'] = model['C'] - LR*model_grads['C']
        model['b1'] = model['b1'] - LR*model_grads['b1']
        model['b2'] = model['b2'] - LR*model_grads['b2']
    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print("Time ", time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    f,H,Z = forward(x, y, model)
    prediction = np.argmax(f)
    if (prediction == y):
        total_correct += 1
print("Test Accuracy: ",total_correct/np.float(len(x_test) ) )
