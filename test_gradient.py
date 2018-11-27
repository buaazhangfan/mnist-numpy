import numpy as np
from layers import MeanBatchNorm
import pytest

def test_MeanBatchNorm_beta():

	T = MeanBatchNorm(1)
	T.beta = np.array([1.05])
	X = np.array([1]).reshape([1,1])
	epsilon = 1e-7

	J = T.forward(X)
	dY = np.array([1])
	_, grad = T.backward(dY)
	grad_beta = grad[0][1]
	
	T.beta = T.beta + epsilon
	J_plus = T.forward(X)
	T.beta = T.beta - 2 * epsilon
	J_minus = T.forward(X)
	num_grad = (J_plus - J_minus) / (2 * epsilon)

	numerator = np.linalg.norm(grad_beta - num_grad)
	denominator = np.linalg.norm(grad_beta) + np.linalg.norm(num_grad)
	difference = numerator / denominator

	assert difference < 1e-7



