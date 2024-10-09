% Filename: NeuralNetwork.m
classdef NeuralNetwork
    properties
        X   % Input matrix
        k   % Size first hidden layer
        X_r % Number of rows of X
        X_c % Number of columns of X
        W1  % Weights for the first layer
        U   % Result of first layer after activation
        W2  % Weights for the second layer
    end
    
    methods
        function obj = NeuralNetwork(X, k, X_r, X_c, W1, W2)
            % Constructor to initialize the properties of the network
            
            obj.X = X;       % Set the input matrix
            obj.k = k;       % Set the size of the first layer
            obj.X_r = X_r;   % Set the number of rows in the input matrix
            obj.X_c = X_c;   % Set the number of columns in the input matrix

            % Initialize W1 if not provided
            if nargin < 5 || isempty(W1)
                obj.W1 = randn(X_c + 1, k);  % Generate W1 randomly
            else
                obj.W1 = W1;  % Use the provided W1
            end

            % Initialize W2 if not provided
            if nargin < 6 || isempty(W2)
                obj.W2 = [];  % W2 will be initialized in secondLayer
            else
                obj.W2 = W2;  % Use the provided W2
            end
        end
        
        function obj = firstLayer(obj, activation_function)
            % Perform the forward pass for the first layer
            % and apply the activation function

            obj.X = obj.addOnes(obj.X);
            Z = obj.X * obj.W1;
            obj.U = activation_function(Z);
        end

        function obj = secondLayer(obj, m)
             % Perform the operations for the second layer

             obj.U = obj.addOnes(obj.U);

             % Initialize random W2 if not already set
             if isempty(obj.W2)
                obj.W2 = randn(obj.k+1, m);
             end
        end

        function eval = evaluateModel(obj, Y, W2)
            % Computes the evaluation metric for the model

            residual = obj.U * W2 - Y;
            % Compute the Frobenius norm squared
            frob_norm_squared = sum(sum(residual.^2)); 
            eval= (1 / (2*obj.X_r)) * frob_norm_squared;

        end
    end
    
    % Private methods 
    methods (Access = private)

        function A = addOnes(obj, A)
            % Add a column of ones to the input matrix A (for bias term)
       
            onesColumn = ones(obj.X_r, 1);
            A = [A onesColumn];
        end
    end
end

