import numpy as np

class Model:
    """
    A simple model class for handling input matrices, activation functions, 
    and hidden layers.
    """
    def __init__(self, X, Y, activation_function, k, seed=None):
        """
        Initializes the model with the input matrix X, activation function, 
        and the number of units in the hidden layer (k).

        Parameters:
        X (numpy.ndarray): Input matrix of shape (n_samples, n_features).
        Y (numpy.ndarray): Output matrix of shape (n_samples, n_output_features).
        activation_function (function): Activation function to be applied element-wise.
        k (int): Number of units in the hidden layer.
        """
        self.X = X  # Input matrix
        self.Y = Y  # Output matrix
        self.activation_function = activation_function  # Activation function
        self.k = k  # Hidden layer size
        self.seed = seed  # Random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)  # Set the seed if provided        

    def hidden_layer(self):
        """
        Computes the hidden layer output using the input matrix, activation function,
        and randomly initialized weights.

        Returns:
        numpy.ndarray: Output of the hidden layer after applying the activation function.
        """
        # Add a column of ones to X (bias term)
        X_bias = np.hstack((self.X, np.ones((self.X.shape[0], 1))))

        # Initialize random weights W1 with shape (n_features + 1, k)
        W1 = np.random.rand(X_bias.shape[1], self.k)

        # Compute Z = X_bias * W1
        Z = np.dot(X_bias, W1)

        # Apply activation function pointwise to Z
        U = self.activation_function(Z)

        return U
    
    def output_layer(self, hidden_output, W2=None):
        """
        Computes the output layer from the hidden layer output by adding a bias term 
        and multiplying with the weights W2. If W2 is not provided, it is initialized randomly.

        Parameters:
        hidden_output (numpy.ndarray): Output from the hidden layer.
        W2 (numpy.ndarray, optional): Weights for the output layer. If not provided, 
                                      they are initialized randomly.

        Returns:
        numpy.ndarray: The final output matrix after multiplying with W2.
        """
        # Add a column of ones to hidden_output (bias term for output layer)
        hidden_output_bias = np.hstack((hidden_output, np.ones((hidden_output.shape[0], 1))))

        # If W2 is not provided, initialize it randomly
        if W2 is None:
            W2 = np.random.rand(hidden_output_bias.shape[1], self.Y.shape[1])

        # Compute output_matrix = hidden_output_bias * W2
        output_matrix = np.dot(hidden_output_bias, W2)

        return output_matrix


    def evaluate_model(self, output_matrix):
        """
        Evaluates the model using the given true output matrix Y and the 
        predicted output matrix output_matrix.

        The evaluation metric is the mean squared error (MSE) scaled by 1 / (2 * n_samples).

        Parameters:
        output_matrix (numpy.ndarray): Predicted output matrix from the model.

        Returns:
        float: The evaluation score calculated as 1/(2*n_samples) * ||output_matrix - Y||_2^2.
        """
        # Compute the difference between output_matrix and Y (element-wise)
        diff = output_matrix - self.Y

        # Compute the Frobenius norm squared (using np.linalg.norm with 'fro' for Frobenius)
        squared_norm = np.linalg.norm(diff, 'fro') ** 2

        # Compute the result as per the formula
        result = 1 / (2 * self.Y.shape[0]) * squared_norm

        return result