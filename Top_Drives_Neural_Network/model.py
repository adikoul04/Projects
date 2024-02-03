from utils import*

### CONSTANTS DEFINING THE MODEL ####
n_x = 11  
n_h = 5
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0000002

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0000002, num_iterations = 505, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->RELU.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "rating" vector (containing the true rating of each vehicle), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initializes parameters dictionary
    parameters = initialize_parameters(n_x, n_h, n_y)
  
    # Gets W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".

        A1, cache1 = linear_activation_forward(X, W1, b1)
        A2, cache2 = linear_activation_forward(A1, W2, b2)

        # Computes cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2)
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1)
        
        # Sets grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Updates parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
      
        # Retrieves W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Prints the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

# Initializes the two layer model
parameters, costs = two_layer_model(data, results, layers_dims = (n_x, n_h, n_y), num_iterations = 505, print_cost=True)

# Asks the user for input about their car. Terminates when the user does not want to enter another car
print("\nNeural network trained. Now you can enter data of your own cars (or make up data) to see their rating in Top Drives\n")
another = True
while(another == True):
    predict(parameters)
    again = input('\nDo you want to enter another car (Yes or No)? ')
    if(again == 'No'):
        print("\nAlright, thanks for using the Top Drives Neural Network!\n")
        another = False
