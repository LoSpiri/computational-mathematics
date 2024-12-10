import numpy as np
import matplotlib.pyplot as plt

class DeflectedSubgradientMethod:
    """
    A class that implements the Deflected Subgradient Method for optimization problems.
    Includes methods to visualize the optimization process.
    """

    def __init__(self, X, Y, delta, rho, R, maxIter, lambda_, seed=None):
        self.X = X
        self.N = X.shape[0]
        self.Y = Y
        self.delta = delta
        self.rho = rho
        self.R = R
        self.maxIter = maxIter
        self.lambda_ = lambda_
        self.y_bar = 0.590  # Target value for relative error computation
        print(self.y_bar)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.W = np.random.rand(X.shape[1], Y.shape[1])
        
        # Vectors to store values for plotting
        self.fx_history = []   # To store f_x values
        self.alpha_history = []  # To store alpha values
        self.gamma_history = []  # To store gamma values
        self.gradient_norm_history = []  # To store ||g|| values
        self.relative_error_history = []  # To store relative error with respect to y_bar

    def compute_deflected_subgradient(self):
        r = 0
        f_ref = self.objective_function(self.W)
        f_bar = f_ref
        f_x = f_bar
        
        x = self.W
        gamma = 1
        beta = 1
        
        for i in range(self.maxIter):
            g = self.__compute_subgradient(x)
            g_norm = np.sqrt(self.__frobenius_norm_squared(g))
             
            #z = min(0.03 * (i // 1000), 1.0) 
            
            if i > 0:
                gamma = self.__compute_gamma(g, d)
                #beta = gamma *(1-z)
                beta = gamma
                d = gamma * g + (1 - gamma) * d
            else:
                d = g
            
            alpha = self.__compute_alpha(f_x, beta, f_ref, d)
            x = x - alpha * d
            
            f_x = self.objective_function(x)
            f_bar = min(f_bar, f_x)
            
            # Update histories for plotting
            self.fx_history.append(f_x)
            self.alpha_history.append(alpha)
            self.gamma_history.append(gamma)
            self.gradient_norm_history.append(g_norm)
            
            relative_error = abs(f_x - self.y_bar) / abs(self.y_bar)
            self.relative_error_history.append(relative_error)
            
            if f_bar <= f_ref - (self.delta / 2):
                f_ref = f_bar
                r = 0
            elif r > self.R:
                self.delta = self.delta * self.rho
                r = 0
            else:
                r = r + alpha * np.sqrt(self.__frobenius_norm_squared(d))
        
        return x

    def objective_function(self, W):
        error_term = (1 / (2 * self.N)) * (self.__frobenius_norm_squared(np.dot(self.X, W) - self.Y))
        regularization_term = self.lambda_ * self.__norm1(W)
        return error_term + regularization_term

    def __compute_subgradient(self, W):
        grad_Lm = np.zeros((self.X.shape[1], self.W.shape[1]))
        for t in range(self.N):
            ht = np.dot(self.X[t, :], W) - self.Y[t, :]
            grad_Lm += np.outer(self.X[t, :], ht)
        grad_Lm /= self.N
        grad_L1 = np.sign(W)
        return grad_Lm + self.lambda_ * grad_L1

    def __compute_alpha(self, f_x, beta, f_ref, d):
        d_frobenius_squared = self.__frobenius_norm_squared(d)
        numerator = f_x - f_ref + self.delta
        alpha = beta * (numerator / d_frobenius_squared)
        min_alpha = 0.001
        return np.clip(alpha, min_alpha, 1)

    def __compute_gamma(self, g, d):
        v = g - d
        v_frobenius_squared = self.__frobenius_norm_squared(v)
        if v_frobenius_squared == 0:
            return 0.5
        dot_product = np.dot(d.flatten(), v.flatten())
        gamma = -dot_product / v_frobenius_squared
        if gamma < 0 or gamma > 1:
            norm_d_squared = self.__frobenius_norm_squared(d)
            if norm_d_squared <= v_frobenius_squared + 2 * dot_product + norm_d_squared:
                gamma = 0
            else:
                gamma = 1
        return gamma

    @staticmethod
    def __frobenius_norm_squared(matrix):
        return np.sum(np.square(matrix))

    @staticmethod
    def __norm1(matrix):
        return np.sum(np.abs(matrix))

    def plot(self, log_scale=False):
        """
        Plots the histories of f_x, ||g||, alpha, and gamma over iterations.
        """
        iterations = range(len(self.fx_history))

        # Plot f_x over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.fx_history, label="Objective Function (f_x)", marker='o')
        plt.xlabel("Iterations")
        plt.ylabel("f_x")
        if log_scale:
            plt.yscale('log')
            plt.title(r"$f_x$ over Iterations (Log Scale)")
        else:
            plt.title(r"$f_x$ over Iterations")
        plt.grid()
        plt.legend()
        plt.show()

        # Plot ||g|| (gradient norm) over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.gradient_norm_history, label="Gradient Norm (||g||)", marker='o', color='orange')
        plt.xlabel("Iterations")
        plt.ylabel("||g||")
        if log_scale:
            plt.yscale('log')
            plt.title("Gradient Norm (||g||) Over Iterations (Log Scale)")
        else:
            plt.title("Gradient Norm (||g||) Over Iterations")
        plt.grid()
        plt.legend()
        plt.show()

        # Plot alpha over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.alpha_history, label="Step Size (alpha)", marker='o', color='green')
        plt.xlabel("Iterations")
        plt.ylabel("Alpha")
        plt.title("Step Size (Alpha) Over Iterations")
        plt.grid()
        plt.legend()
        plt.show()

        # Plot gamma over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.gamma_history, label="Deflection Parameter (gamma)", marker='o', color='red')
        plt.xlabel("Iterations")
        plt.ylabel("Gamma")
        plt.title("Deflection Parameter (Gamma) Over Iterations")
        plt.grid()
        plt.legend()
        plt.show()
        
    def plot_relative_error(self, log_scale=False):
        """
        Plots the relative error between f_x and y_bar over iterations.
        """
        iterations = range(len(self.relative_error_history))
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.relative_error_history, label="Relative Error", marker='o', color='purple')
        plt.xlabel("Iterations")
        plt.ylabel("Relative Error")
        if log_scale:
            plt.yscale('log')
            plt.title("Relative Error Over Iterations (Log Scale)")
        else:
            plt.title("Relative Error Over Iterations")
        plt.grid()
        plt.legend()
        plt.show()
