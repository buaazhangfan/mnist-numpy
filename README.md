# Mnist-numpy with batch normalization and dialted convolution

I implemented the mean-only batch normalization, batch normalization, and the dilated convolution layer in the basic MLP network model

## Task1 Mean-only Batch Normalization
### layers.py
#### class MeanBatchNorm(Layer)

The forward and backward process for mean-only batch normalization is implemented here. Since calculating the gradient for _beta_ and _X_ doesn't neet any cache so there is no difference between train and test process

I initialized the _beta_ to zeros and it has a shape of `(input_dim,)`

In the `main.py` file, set the variable `batch_norm` to `True` and the mean-only batch normalization layer is added after the activation layer. (From the paper it said to add batch norm layer before the activation layer, however I found the result would be better when add it after the activation layer in my practice).

#### Numerical Gradient Check
##### test_gradient.py

In order to check the gradient calculation for parameter _beta_, I wrote a simple test script with _pytest_
For convenience, the input is `(1,)` and the _beta_ was initializaed to a non-zero value. The gradient from upstream `dY` is set to `1`, the `epsilon` is set to `1e-7`. From numerical gradient calculation, just use `beta + epsilon` and `beta - epsilon` to get the output and calculate the gradient from output to _beta_(Since dY is set to 1)

The test is passed.

```
collected 1 item

test_gradient.py .                                                                                                      [100%]

================================================== 1 passed in 0.04 seconds ===================================================
```

##### Result

The network is trained with learning rate `1e-3` for `50` epoches

- The MLP model without batch norm result:

	```
	Epoch:   0, train loss: 1.788, val loss:  1.239, val acc: 0.651
	Epoch:  10, train loss: 0.377, val loss:  0.333, val acc: 0.908
	Epoch:  20, train loss: 0.279, val loss:  0.252, val acc: 0.927
	Epoch:  30, train loss: 0.231, val loss:  0.215, val acc: 0.939
	Epoch:  40, train loss: 0.201, val loss:  0.192, val acc: 0.945
	Epoch:  49, train loss: 0.180, val loss:  0.177, val acc: 0.949
	```
	The result for testing set is:
	
	```
	Test loss: 0.18854853916504768
	Test accuracy: 0.945
	
	```
- The MLP model with batch norm result:
	
	```
	Epoch:   0, train loss: 1.712, val loss:  1.145, val acc: 0.667
	Epoch:  10, train loss: 0.388, val loss:  0.335, val acc: 0.907
	Epoch:  20, train loss: 0.288, val loss:  0.255, val acc: 0.928
	Epoch:  30, train loss: 0.239, val loss:  0.217, val acc: 0.938
	Epoch:  40, train loss: 0.207, val loss:  0.192, val acc: 0.947
	Epoch:  49, train loss: 0.185, val loss:  0.176, val acc: 0.951
	```
	The result for testing set is:
	
	```
	Test loss: 0.18757332088591014
	Test accuracy: 0.9464
	```
The model with mean-only batch normalization performs better than original model, the difference is small is mainly because for mnist classification, the two-layer MLP model is sufficient to train it into a good result.

##### Note

######I also implement the full batch normaliation in the `class BatchNorm(Layer)` 

## Task2 Dilated Convolutional Layer (with extra credit to backward the conv layer)
### layers.py
#### class DilatedConv(Layer)

The dilated conv layer is a little bit complicated to implement in numpy. This class is initialized with five inputs: `n_filter, size_filter, dilated_factor, stride = 1, padding = 2`

1. The `n_filter` is the number of filters in the conv layer
2. The `size_filter` is the size for the basic filter (3x3)
3. The `dilated_factor` is the size to extend the basic filter
4. The `stride` is the step size for filter to process
5. The `padding` is the padding size for original images, since the dialted conv filter is 5x5 and in order to have the same size like input image (28x28), the padding is set to 2.

There are two kinds of parameters in this conv layer, Weights of filters and the biases. The weights are initialized with  Xavier-He initialization and the biases are initialized with zeros.


The shape of weights is `(n_filter, 1, 3, 3)`

The shape of biases is `(n_filter, 1)`

For the forward computation, I implent a construct function to extend the basic 3x3 filter to 5x5 dilated filter. For example, the original 3x3 filter is initialized as:

```
array([[-0.99834786,  0.49452888, -0.6075791 ],
       [-0.08807799,  0.51217256, -0.14430952],
       [ 2.04605076,  0.16820588,  0.04775782]])
```
After the construct function, it will become:

```
array([[-0.99834786,  0.        ,  0.49452888,  0.        , -0.6075791 ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [-0.08807799,  0.        ,  0.51217256,  0.        , -0.14430952],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 2.04605076,  0.        ,  0.16820588,  0.        ,  0.04775782]])
```
This is how I implement the dilated conv layer.

For the backward process, after the `dW_d` gradient is calculated, since only the former 3x3 parameters should be updated, so the destruct function is used to only keep the 3x3 parameters from 5x5 gradient matrix. The process is similiar with the construct function.

##### Notes

For forward calculation, I used the `im2col` method to do matrix manupilation instead of writing a hoop. The `im2col.py` script provided the functions to transform images to columns and columns to images (backward).

The original data set shape is `(batch, 784)`, in order to perform a conv layer, in the forward process, the dataset is reshaped to `(batch, 1, 28, 28)`, and then the output is flatten to `(batch, 784)` to feed into fully connected layer.

I didn't implement the maxpooling layer after the conv layer so the dimension of the output is `(batch, 784)` so the training time is much more longer than original network.

#### Training

The `main_conv.py` is the script for dilated conv network and the learning rate and epoches are the same as task 1.

#### Result

(The result contains the gradient descend in the conv layer)

```
Epoch:   0, train loss: 0.864, val loss:  0.450, val acc: 0.881
Epoch:  10, train loss: 0.196, val loss:  0.184, val acc: 0.950
Epoch:  20, train loss: 0.144, val loss:  0.145, val acc: 0.962
Epoch:  30, train loss: 0.118, val loss:  0.126, val acc: 0.966
Epoch:  40, train loss: 0.100, val loss:  0.114, val acc: 0.970
Epoch:  49, train loss: 0.089, val loss:  0.105, val acc: 0.973
```
The result for testing set is:

```
Test loss: 0.10370359529291456
Test accuracy: 0.97
```
From the result that the model with a dilated conv layer performs much better than fully connected layer and after 50 epoches the test accuracy reached 0.97.
