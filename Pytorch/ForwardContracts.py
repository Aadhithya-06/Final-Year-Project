import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from FBSNNs import FBSNN


class ForwardContracts(FBSNN):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation):
        # Constructor for the BlackScholesBarenblatt class
        # Initializes a new instance with specified parameters for the neural network
        # Inherits from FBSNN (Forward-Backward Stochastic Neural Network)
        # Parameters:
        # Xi: Initial condition
        # T: Time horizon
        # M: Batch size
        # N: Number of time discretization steps
        # D: Dimension of the problem
        # layers: Configuration of the neural network layers
        # mode: Operation mode
        # activation: Activation function for the neural network
        super().__init__(Xi, T, M, N, D, Mm, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):
        # Defines the drift term in the Black-Scholes-Barenblatt equation for a batch
        # t: Batch of current times, size M x 1
        # X: Batch of current states, size M x D
        # Y: Batch of current value functions, size M x 1
        # Z: Batch of gradients of the value function with respect to X, size M x D
        # Returns the drift term for each instance in the batch, size M x 1
        return 0 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  
        # Terminal condition for the Black-Scholes-Barenblatt equation for a batch
        # X: Batch of terminal states, size M x D
        # Returns the terminal condition for each instance in the batch, size M x 1
        # underlying = torch.sum(X, dim=1, keepdim=True)
        # value = torch.maximum(underlying - self.strike * self.D, torch.tensor(0.0)) 
        value = torch.sum(X, dim=1, keepdim=True) - 0.5 * self.D
        return value  # M x 1

    def mu_tf(self, t, X, Y, Z): 
        # Drift coefficient of the underlying stochastic process for a batch
        # Inherits from the superclass FBSNN without modification
        # Parameters are the same as in phi_tf, with batch sizes
        return super().mu_tf(t, X, Y, Z)  # M x D

    def sigma_tf(self, t, X, Y):  
        # Diffusion coefficient of the underlying stochastic process for a batch
        # t: Batch of current times, size M x 1
        # X: Batch of current states, size M x D
        # Y: Batch of current value functions, size M x 1 (not used in this method)
        # Returns a batch of diagonal matrices, each of size D x D, for the diffusion coefficients

        # Assuming sigma is the volatility scalar, in this case, 0.4
        sigma = 0.25

        # L = torch.from_numpy(self.L).float().to(self.device)  # D x D


        # The covariance matrix is sigma^2 times the correlation matrix, so the diffusion matrix in its simplest form
        # would be the identity matrix scaled by sigma, transformed by L to incorporate correlations.
        # However, since L already captures the correlation, we directly use it scaled by sigma for the diffusion.

        # # Create a diffusion matrix that incorporates correlations
        # diffusion_matrix = sigma * L # D x D
       
        return sigma * torch.diag_embed(X) # diffusion_matrix.unsqueeze(0).repeat(self.M, 1, 1) # sigma * torch.diag_embed(X) #    # M x D x D