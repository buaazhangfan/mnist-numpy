import numpy as np
from im2col import *



class Layer(object):
    '''
    Abstract class representing a neural network layer
    '''
    def forward(self, X, train=True):
        '''
        Calculates a forward pass through the layer.

        Args:
            X (numpy.ndarray): Input to the layer with dimensions (batch_size, input_size)

        Returns:
            (numpy.ndarray): Output of the layer with dimensions (batch_size, output_size)
        '''
        raise NotImplementedError('This is an abstract class')

    def backward(self, dY):
        '''
        Calculates a backward pass through the layer.

        Args:
            dY (numpy.ndarray): The gradient of the output with dimensions (batch_size, output_size)

        Returns:
            dX, var_grad_list
            dX (numpy.ndarray): Gradient of the input (batch_size, output_size)
            var_grad_list (list): List of tuples in the form (variable_pointer, variable_grad)
                where variable_pointer and variable_grad are the pointer to an internal
                variable of the layer and the corresponding gradient of the variable
        '''
        raise NotImplementedError('This is an abstract class')

class MeanBatchNorm(Layer):
    '''
    Define the forward and backward computation graph of mean-only batch norm
    '''
    def __init__(self, input_dim):

        self.beta = np.zeros(input_dim)
        self.cache = None

    def forward(self, X, train = True):

        self.N, self.D = X.shape
        mu = np.mean(X, axis = 0)
        x_hat = X - mu
        out = x_hat + self.beta

        return out

    def backward(self, dY):

        dbeta = np.sum(dY, axis = 0)
        N = dY.shape[0]
        dX = (1 - 1 / N) * dY

        return dX, [(self.beta, dbeta)]

class BatchNorm(Layer):
    def __init__(self, input_dim):

        self.beta = np.zeros(input_dim)
        self.gamma = np.random.randn(input_dim)
        self.eps = 1e-5
        self.cache = None

    def forward(self, X, train = True):

        self.N, self.D = X.shape
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        x_mu = X - mean
        inv_var = 1.0 / np.sqrt(var + self.eps)
        x_hat = x_mu * inv_var

        out = self.gamma*x_hat + self.beta

        if train:
            self.cache = (x_mu, inv_var, x_hat, self.gamma)

        return out

    def backward(self, dY):

        N, D = dY.shape
        dxhat = dY * self.cache[3]
        dvar = np.sum((dxhat * self.cache[0] * (-0.5) * (self.cache[1])**3), axis=0)
        dmu = (np.sum((dxhat * -self.cache[1]), axis=0)) + (dvar * (-2.0 / N) * np.sum(self.cache[0], axis=0))
        dx1 = dxhat * self.cache[1]
        dx2 = dvar * (2.0 / N) * self.cache[0]
        dx3 = (1.0 / N) * dmu
        dX = dx1 + dx2 + dx3
        dbeta = np.sum(dY, axis=0)
        dgamma = np.sum(self.cache[2]*dY, axis=0)

        return dX, [(self.beta, dbeta), (self.gamma, dgamma)]

class DilatedConv(Layer):
    '''
    Define the dilated conv layer with dilated_factor = 2
    The 3x3 filter is extended to 5x5
    '''
    def __init__(self, n_filter, size_filter, dilated_factor, stride = 1, padding = 2):
        self.n_filter = n_filter
        self.size_filter = size_filter
        self.dilated_factor = dilated_factor
        self.stride = stride
        self.padding = padding
        self.dilated_factor = dilated_factor
        self.dilated_kernel = self.size_filter + (self.size_filter - 1) * (self.dilated_factor - 1)

        # Construct the dilated conv kernel
        self.W = np.random.randn(n_filter, 1, size_filter, size_filter) / np.sqrt(n_filter / 2)
        self.b = np.zeros((self.n_filter, 1))
        self.params = [self.W, self.b]
        self.dilated_filter = self.construct().reshape([self.n_filter, 1, self.dilated_kernel, self.dilated_kernel])
        self.cache = None

    def construct(self):
        # Extend the 3x3 filter to 5x5 filter with zeros
        pad1 = np.zeros(3)
        pad2 = np.zeros(5).T
        dilated_conv = np.zeros([self.n_filter, 5, 5])
        for i in range(self.n_filter):
            z = np.insert(self.W[i].reshape([3,3]), 1, pad1, 0)
            z = np.insert(z, 3, pad1, 0)
            z = np.insert(z, 1, pad2, 1)
            z = np.insert(z, 3, pad2, 1)
            dilated_conv[i] = z
        return dilated_conv

    def destruct(self, dW_l):
        # Only reserve the 3x3 parameters to update the gradient and reduce the 5x5 filter to 3x3
        dW = np.zeros([self.n_filter, 3, 3])
        for j in range(self.n_filter):
            t = np.delete(dW_l[j].reshape([5, 5]), 1, 0)
            t = np.delete(t, 2, 0)
            t = np.delete(t, 1, 1)
            t = np.delete(t, 2, 1)
            dW[j] = t
        return dW

    def forward(self, X, train = True):

        N, D = X.shape
        X = X.reshape([N, 1, 28, 28])
        # X_col.shape = (n_filter, 100352)
        X_col = im2col_indices(X, 5, 5, padding = self.padding, stride = self.stride)
        W_col = self.dilated_filter.reshape(self.n_filter, -1)

        out = W_col @ X_col + self.b

        out = out.reshape(self.n_filter, 28, 28, N)
        out = out.transpose(3, 0, 1, 2)

        # Flatten
        out = out.reshape(N, -1)

        if train:
            self.cache = (X, X_col, self.dilated_filter, self.b, self.stride, self.padding)
        return out

    def backward(self, dY):

        # reshape dY to (N, n_filter, height, width)
        dY = dY.reshape(dY.shape[0], self.n_filter, 28, 28)
        db = np.sum(dY, axis = (0, 2, 3))
        db = db.reshape(self.n_filter, -1)
        # dY.shape = (n_filter, 100352)
        dY_reshape = dY.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)
        # dW.shape = (n_filter, 25)
        dW_l = dY_reshape @ self.cache[1].T 
        # dW_l.shape = (n_filter, 1, 5, 5)
        dW_l = dW_l.reshape(self.dilated_filter.shape)
        dW = self.destruct(dW_l).reshape([self.n_filter, 1, 3, 3])

        dilated_filter_reshape = self.cache[2].reshape(self.n_filter, -1)
        dX_col = dilated_filter_reshape.T @ dY_reshape
        dX = col2im_indices(dX_col, self.cache[0].shape, 5, 5, padding=self.cache[5], stride=self.cache[4])

        return dX, [(self.W, dW), (self.b, db)]




class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        '''
        Represent a linear transformation Y = X*W + b
            X is an numpy.ndarray with shape (batch_size, input_dim)
            W is a trainable matrix with dimensions (input_dim, output_dim)
            b is a bias with dimensions (1, output_dim)
            Y is an numpy.ndarray with shape (batch_size, output_dim)

        W is initialized with Xavier-He initialization
        b is initialized to zero
        '''
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
        self.b = np.zeros((1, output_dim))

        self.cache_in = None

    def forward(self, X, train=True):
        out = np.matmul(X, self.W) + self.b
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        db = np.sum(dY, axis=0, keepdims=True)
        dW = np.matmul(self.cache_in.T, dY)
        dX = np.matmul(dY, self.W.T)
        return dX, [(self.W, dW), (self.b, db)]

class ReLU(Layer):
    def __init__(self):
        '''
        Represents a rectified linear unit (ReLU)
            ReLU(x) = max(x, 0)
        '''
        self.cache_in = None

    def forward(self, X, train=True):
        if train:
            self.cache_in = X
        return np.maximum(X, 0)

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY * (self.cache_in >= 0), []

class Loss(object):
    '''
    Abstract class representing a loss function
    '''
    def get_loss(self):
        raise NotImplementedError('This is an abstract class')

class SoftmaxCrossEntropyLoss(Loss):
    '''
    Represents the categorical softmax cross entropy loss
    '''

    def get_loss(self, scores, labels):
        '''
        Calculates the average categorical softmax cross entropy loss.

        Args:
            scores (numpy.ndarray): Unnormalized logit class scores. Shape (batch_size, num_classes)
            labels (numpy.ndarray): True labels represented as ints (eg. 2 represents the third class). Shape (batch_size)

        Returns:
            loss, grad
            loss (float): The average cross entropy between labels and the softmax normalization of scores
            grad (numpy.ndarray): Gradient for scores with respect to the loss. Shape (batch_size, num_classes)
        '''
        scores_norm = scores - np.max(scores, axis=1, keepdims=True)
        scores_norm = np.exp(scores_norm)
        scores_norm = scores_norm / np.sum(scores_norm, axis=1, keepdims=True)

        true_class_scores = scores_norm[np.arange(len(labels)), labels]
        loss = np.mean(-np.log(true_class_scores))

        one_hot = np.zeros(scores.shape)
        one_hot[np.arange(len(labels)), labels] = 1.0
        grad = (scores_norm - one_hot) / len(labels)

        return loss, grad

