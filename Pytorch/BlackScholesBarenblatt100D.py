import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from FBSNNs import FBSNN


class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
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
        super().__init__(Xi, T, M, N, D, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):
        # Defines the drift term in the Black-Scholes-Barenblatt equation for a batch
        # t: Batch of current times, size M x 1
        # X: Batch of current states, size M x D
        # Y: Batch of current value functions, size M x 1
        # Z: Batch of gradients of the value function with respect to X, size M x D
        # Returns the drift term for each instance in the batch, size M x 1
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  
        # Terminal condition for the Black-Scholes-Barenblatt equation for a batch
        # X: Batch of terminal states, size M x D
        # Returns the terminal condition for each instance in the batch, size M x 1
        max_values = torch.relu(X-0.5)
        return torch.sum(max_values, 1, keepdim=True)  # M x 1

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

        # L = torch.from_numpy(self.L).float().to(self.device)  # D x D

        # Assuming sigma is the volatility scalar, in this case, 0.4
        sigma = 0.4

        # The covariance matrix is sigma^2 times the correlation matrix, so the diffusion matrix in its simplest form
        # would be the identity matrix scaled by sigma, transformed by L to incorporate correlations.
        # However, since L already captures the correlation, we directly use it scaled by sigma for the diffusion.

        # # Create a diffusion matrix that incorporates correlations
        # diffusion_matrix = sigma * L  # This needs to be adjusted based on how you use L and sigma in your model's context
        # diffusion_matrix.unsqueeze(0).repeat(self.M, 1, 1)
        return sigma * torch.diag_embed(X)  # M x D x D


def u_exact(T, t, X):
    # Calculates the exact solution for the Black Scholes Barenblatt equation
    # Parameters:
    # T: The terminal time, a vector of size (N+1) x 1. Represents the final time in the time discretization
    # t: The current time, a vector of size (N+1) x 1. Represents the time steps in the time discretization
    # X: The current state, an array of size (N+1) x D. Represents the state variables at each time step
    
    r = 0.05         # Represents the risk-free interest rate
    sigma_max = 0.4  # Represents the maximum volatility

    # The exact solution is calculated using an exponential term and a summation term
    # The exponential term accounts for the time value of money and volatility
    # The summation term represents the square of the state variables summed across the D dimensions
    # The solution is computed for each time step and state, resulting in a vector of size (N+1) x 1
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.maximum(X-0.5, 0).sum(axis=1,keepdims=True)  # (N+1) x 1
