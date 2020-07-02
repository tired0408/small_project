# encoding:utf-8
import numpy as np 
import matplotlib.pyplot as plt 
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

def layer_sizes(X,Y):
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]
    return (n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)

    W1=np.random.randn(n_h,n_x)
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)
    b2=np.zeros((n_y,1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters={"W1":W1,
    "b1":b1,
    "W2":W2,
    "b2":b2}

    return parameters

def forward_propagation(X,parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)

    assert(A2.shape==(1,X.shape[1]))

    cache={"Z1":Z1,
    "A1":A1,
    "Z2":Z2,
    "A2":A2}

    return A2,cache 

def compute_cost(A2,Y,parameters):
    m=Y.shape[1]

    logprobs=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))
    cost=-(1.0/m)*np.sum(logprobs)

    cost=np.squeeze(cost)

    assert(isinstance(cost,float))

    return cost

def backward_propagation(parameters,cache,X,Y):
    m=X.shape[1]

    W1=parameters["W1"]
    W2=parameters["W2"]

    A1=cache["A1"]
    A2=cache["A2"]

    dZ2=A2-Y
    dW2=1.0/m*np.dot(dZ2,A1.T)
    db2=1.0/m*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1=1.0/m*np.dot(dZ1,X.T)
    db1=1.0/m*np.sum(dZ1,axis=1,keepdims=True)

    grads={"dW1":dW1,
    "db1":db1,
    "dW2":dW2,
    "db2":db2}

    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    dW1=grads["dW1"]
    db1=grads["db1"]
    dW2=grads["dW2"]
    db2=grads["db2"]

    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2

    parameters={"W1":W1,
    "b1":b1,
    "W2":W2,
    "b2":b2}

    return parameters

def nn_model(X,Y,n_h,num_iterations=10000,print_cost=False):
    np.random.seed(3)
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]

    parameters=initialize_parameters(n_x,n_h,n_y)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    for i in range(0,num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=1.2)

        if print_cost and i%1000 == 0:
            print("Cost after iteration %i: %f" %(i,cost))
    
    return parameters

def predict(parameters,X):
    A2,cache=forward_propagation(X,parameters)
    predictions=(A2>0.5)
    
    return predictions

X,Y=load_planar_dataset()

plt.figure(figsize=(32, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(4, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.flatten())
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

plt.show()

# Build a model with a n_h-dimensional hidden layer
#parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
#plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.flatten())
#plt.title("Decision Boundary for hidden layer size " + str(4))
#plt.show()

#predictions = predict(parameters, X)
#print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

#parameters, X_assess = predict_test_case()
#predictions = predict(parameters, X_assess)

#X_assess, Y_assess = nn_model_test_case()
#parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)

#parameters, grads = update_parameters_test_case()
#parameters = update_parameters(parameters, grads)

#parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#grads = backward_propagation(parameters, cache, X_assess, Y_assess)

#A2,Y_assess,parameters=compute_cost_test_case()

#X_assess,Y_assess=layer_sizes_test_case()
#(n_x,n_h,n_y)=layer_sizes(X_assess,Y_assess)

#n_x,n_h,n_y=initialize_parameters_test_case()
#parameters = initialize_parameters(n_x, n_h, n_y)

#X_assess,parameters=forward_propagation_test_case()
#A2,cache=forward_propagation(X_assess,parameters)




