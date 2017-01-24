#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Created on Fri Jan 13 18:37:13 2017
    
    @author: christopherhamblin
    """
"""
    
    Simple example of building a nueral net with one hidden layer in python
    
    """




#import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#Generate data sets to train network on


np.random.seed(0)
n_samples=200   #number of data points

#Create data of noisey shapes from sklearn.datasets
Xmoon, ymoon = datasets.make_moons(n_samples=n_samples, noise=.2)
Xcirc, ycirc = datasets.make_circles(n_samples=n_samples, noise=.2, factor=.5)
Xblob, yblob = datasets.make_blobs(n_samples=n_samples)

#function for creating more simple data in higher dimensions
def create_data(dim=2, n_samples=100, func='cube'):
    X=np.random.rand(n_samples, dim)
    if func=='cube':
        y=np.zeros(n_samples)
        for i in xrange(n_samples):
            for j in xrange(dim):
                if X[i,j]>.75 or X[i,j]<.25:
                    y[i]=1
                    break
    else:
        print ('Error: undefined function')
        return
    if dim==2:
        plt.scatter(X[:,0],X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    if dim==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], s=20, c=y, cmap=plt.cm.Spectral)
    return X, y



#Function to plot data over colored regions showing the decision boundary of the net
def plot_decision_boundary(pred_func, X, y, h=.1):   #h is the grain size
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    if type(Z)==dict:
        vector=Z['vector']
        onehot=Z['onehot']
        vectorr=vector.reshape(xx.shape)
    #Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, vectorr, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()
    return

#takes a vector of different values and turns it into a matrix of 0's and 1's,
# [1,2,3,1] -->   [[1,0,0],[0,1,0],[0,0,1],[1,0,0]]
def one_hot_encoder(y):
    uni=np.unique(y)
    Y=np.zeros((y.size,uni.size))
    c1=0
    for i in y:
        c2=0
        for j in uni:
            if i==j:
                Y[c1,c2]=1
                break
            c2+=1
        c1+=1
    return Y

#Activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):    #Derivative of sigmoid function
    return x*(1.0-x)

def rectlin(x):
    return max(0,x)

def predict(model, test):    #test is the matrix of new input data we want to predict
    
    prediction={}
    
    #get weights from model
    W1=model['W1']
    W2=model['W2']
    b1=model['b1']
    b2=model['b2']
    out_dim=model['out_dim']
    
    #feed forward
    s1 = test.dot(W1) + b1
    X1 = np.tanh(s1)
    s2 = X1.dot(W2) + b2
    out=np.exp(s2)/np.sum(np.exp(s2), axis=1, keepdims=True) #output of model
    
    #Prediction of winning class, one hot encoded
    vector=np.argmax(out,axis=1)
    onehot=np.zeros((len(test),out_dim))
    for i in xrange(np.size(vector)):
        onehot[i,vector[i]]=1
    
    prediction={'vector':vector,'onehot':onehot}
    return prediction


def error_function(y, model): #mean error
    out=model['out']
    loss = np.sum(-np.log(out[range(n_samples), y]))/n_samples
    # Add regulatization term to loss (optional)
    return loss




def build_model(X, y, hid_dim=3, act_func='tanh', grad_passes=3000, learning_rate=.01, regularization=.01, print_loss=True, print_boundary=True):
    #X is input matrix, y is labels (encoded as an integer vector), hid_dim is number of hidden nodes,
    #act_func is the activation function on the hidden layer, grad_passes is number of times we run through gradient descent to train weights
    np.random.seed(0)
    model={} #The model will return weights and biases trained with gradient descent
    #a predition function will then use this model to predict new data
    
    #Set Net Parameters
    in_dim=X.shape[1]     # input layer dimensions
    out_dim=np.size(np.unique(y)) # output layer dimensions
    
    
    #One-hot encode output
    Y=one_hot_encoder(y)
    
    
    #Set initial weights to random values, we'll have to learn these
    W1=np.random.randn(in_dim,hid_dim)/np.sqrt(out_dim)  # weight matrix from input to hidden layer
    W2=np.random.randn(hid_dim,out_dim)/np.sqrt(hid_dim) # weight matrix from hidden layer to output layer
    b1=np.zeros((1,hid_dim))       #0's bias vector from input layer to hidden layer
    b2=np.zeros((1,out_dim))       #0's bias vector from hidden layer to output layer
    #print "W1: %s" %W1
    #print "b1: %s" %b1
    #print "W2: %s" %W2
    #print "b2: %s" %b2
    
    #train net with batch gradient descent
    for i in xrange(grad_passes+1):
        
        if act_func=='tanh':
            #forward Propogation
            s1=X.dot(W1)+b1       #pass input through first weight matrix
            X1=np.tanh(s1)      #pass hidden layer input through activation function
            s2=X1.dot(W2)+b2    #pass hidden layer output through second weight matrix
            #print "s1: %s" %s1
            #print "X1: %s" %X1
            #print "s2: %s" %s2
            #Pass output layer input through softmax to normalize final result to a probability dist
            s2exp= np.exp(s2)
            out=s2exp/ np.sum(s2exp, axis=1, keepdims=True)
            #print "s2exp: %s" %s2exp
            #print "out: %s" %out
            
            #Back propogation  (calculate gradients at each node to minimize cross entropy cost function)
            #CEcost(z2norm,Y)=-sumoversamples(sumoverclasses(Y*log(z2norm)))
            d2=out-Y   #see http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/ for derivation
            #print "d2 %s:" %d2
            #print "Y %s:" %Y
            dW2=(X1.T).dot(d2)
            db2=np.sum(d2, axis=0, keepdims=True)
            d1=d2.dot(W2.T) * (1 - np.power(X1, 2))   #d(l-1)=(1-(x(l-1)^2)*sum(w(l)d(l)) for tanh
            dW1=(X.T).dot(d1)
            db1=np.sum(d1,axis=0)
            
            
            #add regularization
            dW1+=regularization*W1
            dW2+=regularization*W2
            
            #Update weight matrices
            W1-= learning_rate*dW1
            W2-= learning_rate*dW2
            b1-= learning_rate*db1
            b2-= learning_rate*db2
        
        elif act_func=='sigmoid':
            
            #Forward Propogation
            s1=X.dot(W1)+b1
            X1=sigmoid(s1)
            s2=X1.dot(W2)+b2
            #Softmax
            s2exp=np.exp(s2)
            out=s2exp/np.sum(s2exp, axis=1, keepdims=True)
            
            
            #Back Propogation
            d2=out-Y
            dW2=(X1.T).dot(d2)
            db2=np.sum(d2, axis=0, keepdims=True)
            d1=d2.dot(W2.T) * X1 * (1-X1)
            dW1=(X.T).dot(d1)
            db1=np.sum(d1,axis=0)
        
        else:
            print "Error: undefined Activation Function"
            break

        #Update weight matrices
        W1-= learning_rate*dW1
        W2-= learning_rate*dW2
        b1-= learning_rate*db1
        b2-= learning_rate*db2
        
        
        
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'in_dim':in_dim,'out_dim':out_dim,'hid_dim':hid_dim, 'out':out}
        
        if print_loss and i % 200 == 0:
            print ("Loss after pass %i: %f" %(i, error_function(y, model)))

if print_boundary and i % 500 == 0:
    plot_decision_boundary(lambda x: predict(model, x), Xmoon, ymoon)
    
    
    return model


model = build_model(X=Xmoon, y=ymoon, act_func='tanh')














