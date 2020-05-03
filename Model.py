import numpy as np
import DeepNN
import matplotlib.pyplot as plt
from utils import one_hot, N_CLASSES


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    #np.random.seed(1)
    costs = []
    Y = one_hot(Y, N_CLASSES)

    # Parameters initialization.
    parameters = DeepNN.initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = DeepNN.L_model_forward(X, parameters, hidden_layers_activation_fn="sigmoid")

        # Compute cost.
        cost = DeepNN.compute_cost(AL, Y)

        # Backward propagation.
        grads = DeepNN.L_model_backward(AL, Y, caches, hidden_layers_activation_fn="sigmoid")

        # Update parameters.
        parameters = DeepNN.update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X_test, parameters):
    """
    Generate array of predicted labels for the input dataset

    Arguments:
    X -- input data of shape (number of features, number of examples)

    Returns:
    predicted labels of shape (1, n_samples)
    """
    AL, caches = DeepNN.L_model_forward(X_test, parameters, hidden_layers_activation_fn="sigmoid")

    return np.argmax(AL, axis=0).T
