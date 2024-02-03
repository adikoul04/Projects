import numpy as np
import copy
import math
from data_processed import data_formatted as data
from data_processed import results_formatted as results

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(1)
    
    # Initializes W1, W2, b2, and b2
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters   

def linear_forward(A, W, b):
    """
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    # Initializes Z
    Z = np.dot(W, A)+b

    cache = (A, W, b)
    
    return Z, cache

def relu(Z):
    """
    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # converting dz to a correct object.
    
    # When z <= 0, sets dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def linear_activation_forward(A_prev, W, b):
    """
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    
    # Does forward propagation and computes activation
    Z, linear_cache = linear_forward(A_prev,W,b)
    A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def compute_cost(AL, Y):
    """
    Arguments:
    AL -- vector corresponding to the label predictions, shape (1, number of examples)
    Y -- true "rating" vector, shape (1, number of examples)

    Returns:
    cost -- cost of the outputs compared to the true ratings with the current parameters
    """
    
    m = Y.shape[1]

    # Computes loss
    cost = (1./m) * np.sum(np.square(Y-AL), axis=1)
    
    cost = np.squeeze(cost)      # To make sure the cost's shape is what we expect
    cost = math.sqrt(cost)
    return cost

def linear_backward(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    # Calculates gradients
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache):
    """
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    # Calculates gradients
    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def update_parameters(params, grads, learning_rate):
    """
    Arguments:
    params -- python dictionary containing parameters 
    grads -- python dictionary containing gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = params.get("W" + str(l+1))-grads.get("dW" + str(l+1))*learning_rate
        parameters["b" + str(l+1)] = params.get("b" + str(l+1))-grads.get("db" + str(l+1))*learning_rate
        
    return parameters

def predict(params):
    """
    Arguments:
    params -- python dictionary containing parameters 

    Returns:
    Tells the user the predicted rating of the car based on the entered statistics
    """

    # Asks the user for statistics of the car
    name = input('Enter name of car: ')
    year = input('What year was it made: ')
    speed = input('What is its top speed (mph): ')
    accel = input('What is its 0-60 time (sec): ')
    handle = input('What is the handling of the car: ')
    peakpwr = input('What is its peak power: ')
    peaktrq = input('What is its peak torque: ')
    tracanswer = input('Does it have traction control (Yes or No): ')
    if(tracanswer=='Yes'):
        trac = 1.
    else:
        trac = 0.
    absanswer = input('Does it have ABS (Yes or No): ')
    if(absanswer=='Yes'):
        abs = 1.
    else:
        abs = 0.
    groundanswer = input('What is its ground clearance (Low, Medium, or High): ')
    if(groundanswer=='Low'):
        ground = 1.
    elif(groundanswer=='Medium'):
        ground = 2.
    elif(groundanswer=='High'):
        ground = 3.
    driveanswer = input('What drivetrain does it have (FWD, RWD, or 4WD): ')
    if(driveanswer=='FWD'):
        drive = 1.
    elif(driveanswer=='RWD'):
        drive = 2.
    elif(driveanswer=='4WD'):
        drive = 3.
    tireanswer = input('Which tires does it have (Slick, Performance, Standard, All-Surface, or Off-Road): ')
    if(tireanswer=='Slick'):
        tire = 1.
    elif(tireanswer=='Performance'):
        tire = 2.
    elif(tireanswer=='Standard'):
        tire = 3.
    elif(tireanswer=='All-Surface'):
        tire = 4.
    elif(tireanswer=='Off-Road'):
        tire = 5.

    # Uses the entered statistics to create a vector to use in forward propagation
    input_layer = [[float(year)],
                   [float(speed)],
                   [float(accel)],
                   [float(handle)],
                   [float(peakpwr)],
                   [float(peaktrq)],
                   [trac],
                   [abs],
                   [ground],
                   [drive],
                   [tire]]
    
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]


    # Calculates the predicted rating using linear activation
    A1, cache1 = linear_activation_forward(input_layer, W1, b1)
    A2, cache2 = linear_activation_forward(A1, W2, b2)

    answer = np.squeeze(A2)

    # Tells the user the rating of their car
    print("\nThe "+name+" would have an RQ cost of about "+str(answer))
    if(answer<7):
        print("This car would recieve an F rating")
    elif(answer<11):
         print("This car would recieve an E rating")
    elif(answer<15):
         print("This car would recieve a D rating")
    elif(answer<19):
         print("This car would recieve a C rating")
    elif(answer<23):
         print("This car would recieve a B rating")
    elif(answer<17):
         print("This car would recieve an A rating")
    else:
        print("This car would recieve an S rating")
