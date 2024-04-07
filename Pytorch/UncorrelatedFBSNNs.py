import numpy as np
from abc import ABC, abstractmethod
import time

import torch
import torch.nn as nn
import torch.optim as optim

from Models import *


class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation):
        # Constructor for the FBSNN class
        # Initializes the neural network with specified parameters and architecture
        
        # Parameters:
        # Xi: Initial condition (numpy array) for the stochastic process
        # T: Terminal time
        # M: Number of trajectories (batch size)
        # N: Number of time snapshots
        # D: Number of dimensions for the problem
        # Mm: Number of discretization points for the SDE
        # layers: List indicating the size of each layer in the neural network
        # mode: Specifies the architecture of the neural network (e.g., 'FC' for fully connected)
        # activation: Activation function to be used in the neural network

        # Check if CUDA is available and set the appropriate device (GPU or CPU)
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        # Initialize the initial condition, convert it to a PyTorch tensor, and send to the device
        self.Xi = torch.from_numpy(Xi).float().to(self.device)  # initial point
        self.Xi.requires_grad = True

        # Store other parameters as attributes of the class.
        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.Mm = Mm  # number of discretization points for the SDE
        self.strike = 1 * self.D  # strike price
        # self.L = self.generate_cholesky()  # Cholesky decomposition of the correlation matrix

        self.mode = mode  # architecture of the neural network
        self.activation = activation  # activation function        # Initialize the activation function based on the provided parameter
        if activation == "Sine":
            self.activation_function = Sine()
        elif activation == "ReLU":
            self.activation_function = nn.ReLU()

        # Initialize the neural network based on the chosen mode
        if self.mode == "FC":
            # Fully Connected architecture
            self.layers = []
            for i in range(len(layers) - 2):
                self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
                self.layers.append(self.activation_function)
            self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))
            self.model = nn.Sequential(*self.layers).to(self.device)

        elif self.mode == "NAIS-Net":
            # NAIS-Net architecture
            self.model = Resnet(layers, stable=True, activation=self.activation_function).to(self.device)
        elif self.mode == "Resnet":
            # Residual Network architecture
            self.model = Resnet(layers, stable=False, activation=self.activation_function).to(self.device)
        elif self.mode == "Verlet":
            # Verlet Network architecture
            self.model = VerletNet(layers, activation=self.activation_function).to(self.device)
        elif self.mode == "SDEnet":
            # SDE Network architecture
            self.model = SDEnet(layers, activation=self.activation_function).to(self.device)

        # Apply a custom weights initialization to the model.
        self.model.apply(self.weights_init)

        # Initialize lists to record training loss and iterations.
        self.training_loss = []
        self.iteration = []


    def weights_init(self, m):
        # Custom weight initialization method for neural network layers
        # Parameters:
        # m: A layer of the neural network

        if type(m) == nn.Linear:
            # Initialize the weights of the linear layer using Xavier uniform initialization
            torch.nn.init.xavier_uniform_(m.weight)

    def net_u(self, t, X):  # M x 1, M x D
        # Computes the output of the neural network and its gradient with respect to the input state X
        # Parameters:
        # t: A batch of time instances, with dimensions M x 1
        # X: A batch of state variables, with dimensions M x D

        # Concatenate the time and state variables along second dimension
        # to form the input for the neural network
        input = torch.cat((t, X), 1)  

        # Pass the concatenated input through the neural network model
        # The output u is a tensor of dimensions M x 1, representing the value function at each input (t, X)
        u = self.model(input)  # M x 1

        # Compute the gradient of the output u with respect to the state variables X
        # The gradient is calculated for each input in the batch, resulting in a tensor of dimensions M x D
        Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(u), 
                                allow_unused=True, retain_graph=True, create_graph=True)[0]

        return u, Du

    def Dg_tf(self, X):  # M x D
        # Calculates the gradient of the function g with respect to the input X
        # Parameters:
        # X: A batch of state variables, with dimensions M x D

        g = self.g_tf(X)  # M x 1

        # Now, compute the gradient of g with respect to X
        # The gradient is calculated for each input in the batch, resulting in a tensor of dimensions M x D
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), 
                                allow_unused=True, retain_graph=True, create_graph=True)[0] 

        return Dg


    def loss_function(self, t, W, Xi):
        # Calculates the loss for the neural network
        # Parameters:
        # t: A batch of time instances, with dimensions M x (N+1) x 1
        # W: A batch of Brownian motion increments, with dimensions M x (N+1) x D
        # Xi: Initial state, with dimensions 1 x D

        loss = 0  # Initialize the loss to zero.
        X_list = []  # List to store the states at each time step.
        Y_list = []  # List to store the network outputs at each time step.

        # Initial time and Brownian motion increment.
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        # Initial state for all trajectories
        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)  # M x D
        Y0, Z0 = self.net_u(t0, X0)  # Obtain the network output and its gradient at the initial state

        # Store the initial state and the network output
        X_list.append(X0)
        Y_list.append(Y0)

        # Iterate over each time step
        for n in range(0, self.N):
            # Next time step and Brownian motion increment
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            # Compute the next state using the Euler-Maruyama method
            X1 = (1 + 0.05 * (t1 - t0)) * X0 + 0.4 * X0 * (W1 - W0)
            
            # Compute the predicted value (Y1_tilde) at the next state
            Y1_tilde = Y0 + 0.05 * Y0 * (t1 - t0) + torch.sum(
                Z0 * (W1 - W0), dim=1, keepdim=True)

            # Obtain the network output and its gradient at the next state
            Y1, Z1 = self.net_u(t1, X1)

            # Add the squared difference between Y1 and Y1_tilde to the loss
            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            # Update the variables for the next iteration
            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1

            # Store the current state and the network output
            X_list.append(X0)
            Y_list.append(Y0)

        # Add the terminal condition to the loss: 
        # the difference between the network output and the target at the final state
        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        # Add the difference between the network's gradient and the gradient of g at the final state
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))

        # Stack the states and network outputs for all time steps
        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        # Return the loss and the states and outputs at each time step
        # The final element returned is the first element of the network output, for reference or further use
        return loss, X, Y, Y[0, 0, 0]


    def fetch_minibatch(self):  # Generate time + a Brownian motion
        # Generates a minibatch of time steps and corresponding Brownian motion paths

        T = self.T  # Terminal time
        M = self.M  # Number of trajectories (batch size)
        N = self.N  # Number of time snapshots
        D = self.D  # Number of dimensions

        # Initialize arrays for time steps and Brownian increments
        Dt = np.zeros((M, N + 1, 1))  # Time step sizes for each trajectory and time snapshot
        DW = np.zeros((M, N + 1, D))  # Brownian increments for each trajectory, time snapshot, and dimension

        # Calculate the time step size
        dt = T / N

        # Populate the time step sizes for each trajectory and time snapshot (excluding the initial time)
        Dt[:, 1:, :] = dt

        # Generate Brownian increments for each trajectory and time snapshot
        DW_uncorrelated = np.sqrt(dt) * np.random.normal(size=(M, N, D))
        DW[:, 1:, :] = DW_uncorrelated # np.einsum('ij,mnj->mni', self.L, DW_uncorrelated) # Apply Cholesky matrix to introduce correlations

        # Cumulatively sum the time steps and Brownian increments to get the actual time values and Brownian paths
        t = np.cumsum(Dt, axis=1)  # Cumulative time for each trajectory and time snapshot
        W = np.cumsum(DW, axis=1)  # Cumulative Brownian motion for each trajectory, time snapshot, and dimension

        # Convert the numpy arrays to PyTorch tensors and transfer them to the configured device (CPU or GPU)
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        # Return the time values and Brownian paths.
        return t, W

    def train(self, N_Iter, learning_rate):
        # Train the neural network model.
        # Parameters:
        # N_Iter: Number of iterations for the training process
        # learning_rate: Learning rate for the optimizer

        # Initialize an array to store temporary loss values for averaging
        loss_temp = np.array([])

        # Check if there are previous iterations and set the starting iteration number
        previous_it = 0
        if self.iteration != []:
            previous_it = self.iteration[-1]

        # Set up the optimizer (Adam) for the neural network with the specified learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Record the start time for timing the training process
        start_time = time.time()
        # Training loop
        for it in range(previous_it, previous_it + N_Iter):
            # if it >= 4000:
            #     self.N = int(np.ceil(self.Mm ** (int(it / 4000) + 1)))
            # elif it < 4000:
            #     self.N = int(np.ceil(self.Mm))

            # Zero the gradients before each iteration
            self.optimizer.zero_grad()

            # Fetch a minibatch of time steps and Brownian motion paths
            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D

            # Compute the loss for the current batch
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)

            # Perform backpropagation
            self.optimizer.zero_grad()  # Zero the gradients again to ensure correct gradient accumulation
            loss.backward()  # Compute the gradients of the loss w.r.t. the network parameters
            self.optimizer.step()  # Update the network parameters based on the gradients

            # Store the current loss value for later averaging
            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

            # Print the training progress every 100 iterations
            if it % 100 == 0:
                elapsed = time.time() - start_time  # Calculate the elapsed time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                    (it, loss, Y0_pred, elapsed, learning_rate))
                start_time = time.time()  # Reset the start time for the next print interval

            # Record the average loss and iteration number every 100 iterations
            if it % 100 == 0:
                self.training_loss.append(loss_temp.mean())  # Append the average loss
                loss_temp = np.array([])  # Reset the temporary loss array
                self.iteration.append(it)  # Append the current iteration number

        # Stack the iteration and training loss for plotting
        graph = np.stack((self.iteration, self.training_loss))

        # Return the training history (iterations and corresponding losses)
        return graph


    def predict(self, Xi_star, t_star, W_star):
        # Predicts the output of the neural network
        # Parameters:
        # Xi_star: The initial state for the prediction, given as a numpy array
        # t_star: The time steps at which predictions are to be made
        # W_star: The Brownian motion paths corresponding to the time steps

        # Convert the initial state (Xi_star) from a numpy array to a PyTorch tensor
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True

        # Compute the loss and obtain predicted states (X_star) and outputs (Y_star) using the trained model
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)

        # Return the predicted states and outputs
        # These predictions correspond to the neural network's estimation of the state and output at each time step
        return X_star, Y_star

    def generate_cholesky(self):
        # Variances of the individual assets
        rho = 0.5 

        # Create an identity matrix for the diagonal
        correlation_matrix = np.eye(self.D)

        # Set off-diagonal elements to rho
        correlation_matrix[correlation_matrix == 0] = rho


        # Check if the correlation matrix is valid
        if not np.allclose(correlation_matrix, correlation_matrix.T):
            raise ValueError("Correlation matrix is not symmetric.")
        if np.any(np.linalg.eigvalsh(correlation_matrix) < 0):
            raise ValueError("Correlation matrix is not positive semi-definite.")

        # Standard deviations (square roots of variances)
        L = np.linalg.cholesky(correlation_matrix)

        return L

    def save_model(self, file_name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_loss': self.training_loss,
            'iteration': self.iteration
        }, file_name)
    
    def load_model(self, file_name):
        checkpoint = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_loss = checkpoint['training_loss']
        self.iteration = checkpoint['iteration']

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        # Abstract method for defining the drift term in the SDE
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Z: Gradient of the function with respect to state variables, size M x D
        # Expected return size: M x 1
        pass

    @abstractmethod
    def g_tf(self, X):  # M x D
        # Abstract method for defining the terminal condition of the SDE
        # Parameter:
        # X: Terminal state variables, size M x D
        # Expected return size: M x 1
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        # Abstract method for defining the drift coefficient of the underlying stochastic process
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Z: Gradient of the function with respect to state variables, size M x D
        # Default implementation returns a zero tensor of size M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        # Abstract method for defining the diffusion coefficient of the underlying stochastic process
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Default implementation returns a diagonal matrix of ones of size M x D x D
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)  # M x D x D