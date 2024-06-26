import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from XVAFBSNNs import XVAFBSNN


class XVA(XVAFBSNN):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation, model):
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
        super().__init__(Xi, T, M, N, D, Mm, layers, mode, activation, model)

    def phi_tf(self, t, C, Y, Z):
        # Defines the drift term in the Black-Scholes-Barenblatt equation for a batch
        # t: Batch of current times, size M x 1
        # C: Batch of predicted prices, size M x D
        # Y: Batch of current value functions, size M x 1
        # Z: Batch of gradients of the value function with respect to X, size M x D
        # Returns the XVA term for each instance in the batch, size M x 1
        rate = 0.02 # Risk-free interest rate
        r_fl = 0.04 # Funding rate for long position
        r_fb = 0.04 # Funding rate for short position
        r_cl = 0.00 # Collateral rate for long position
        r_cb = 0.00 # Collateral rate for short position
        R_C = 1.0 # Recovery rate for counterparty
        R_B = 1.0 # Recovery rate for bank
        alpha = 0 # Alpha for collateral
        collateral = 0 # Collateral
        intensityC = 0 # Intensity for counterparty
        intensityB = 0 # Intensity for bank

        discount = (rate + intensityC + intensityB) * Y
        # cva = (1-R_C) * torch.maximum(collateral-C, torch.tensor(0.0)) * intensityC
        # dva = (1-R_B) * torch.maximum(C-collateral, torch.tensor(0.0)) * intensityB
        fva = (r_fl - rate) * torch.maximum(C-Y-collateral, torch.tensor(0.0)) - (r_fb - rate) * torch.maximum(collateral+Y-C, torch.tensor(0.0))
        # colva = (r_cl - rate) * torch.maximum(collateral, torch.tensor(0.0)) - (r_cb - rate) * torch.maximum(-collateral, torch.tensor(0.0))

        return -fva + discount #cva - dva + discount #cva - dva - fva - colva + discount  # M x 1

    def g_tf(self, C):  
        # Terminal condition for the Black-Scholes-Barenblatt equation for a batch
        # C: Batch of predicted prices, size M x D
        # Returns the terminal condition for each instance in the batch, size M x 1
        return 0 * C


    def sigma_tf(self, t, C, Y):  
        # Diffusion coefficient of the underlying stochastic process for a batch
        # t: Batch of current times, size M x 1
        # C: Batch of predicted prices, size M x D
        # Y: Batch of current value functions, size M x 1 (not used in this method)
        # Returns a batch of diagonal matrices, each of size D x D, for the diffusion coefficients
        # Each matrix is scaled by 0.4 times the corresponding state in X
        return 0.25 * torch.diag_embed(C)  # M x D x D