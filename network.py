import numpy as np

class Network(object):
    '''
    Represents a neural network with any combination of layers
    '''
    def __init__(self, learning_rate):
        '''
        Returns a new empty neural network with no layers or loss

        Args:
            learning_rate (float): Learning rate to be used for minibatch SGD
        '''
        self.lr = learning_rate
        self.layers = []
        self.loss = None

    def add_layer(self, layer):
        '''
        Adds a layer to the network in a sequential manner. 
        The input to this layer will be the output of the last added layer 
        or the initial inputs to the networks if this is the first layer added.

        Args:
            layer (Layer): An instantiation of a class that extends Layer
        '''
        self.layers.append(layer)

    def set_loss(self, loss):
        '''
        Sets the loss that the network uses for training

        Args:
            loss (Loss): An instantiation of a class that extends Loss
        '''
        self.loss = loss
    
    def predict(self, inputs, train=False):
        '''
        Calculates the output of the network for the given inputs.

        Args:
            inputs (numpy.ndarray): Inputs to the network

        Returns:
            (numpy.ndarray): Outputs of the last layer of the network.
        '''
        scores = inputs
        for layer in self.layers:
            scores = layer.forward(scores, train=train)
        return scores
    
    def train(self, inputs, labels):
        '''
        Calculates the loss of the network for the given inputs and labels
        Performs a gradient descent step to minimize the loss

        Args:
            inputs (numpy.ndarray): Inputs to the network
            labels (numpy.ndarray): Int representation of the labels (eg. the third class is represented by 2)

        Returns:
            (float): The loss before updating the network
        '''
        vars_and_grads = []

        # Forward pass
        scores = self.predict(inputs, train=True)

        # Backward pass
        loss, grad = self.loss.get_loss(scores, labels)
        for layer in reversed(self.layers):
            grad, layer_var_grad = layer.backward(grad)
            vars_and_grads += layer_var_grad

        # Gradient descent update:
        for var_grad in vars_and_grads:
            var, grad = var_grad
            var -= self.lr * grad
        
        return loss