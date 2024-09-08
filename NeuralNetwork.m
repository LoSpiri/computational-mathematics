% Filename: NeuralNetwork.m
classdef NeuralNetwork
    properties
        X   % Input matrix
        k   % Size first hidden layer
        X_r % Number of rows of X
        X_c % Number of columns of X
        W1  % Weights for the first layer
        U   % Result of first layer
        W2  % Weights for the second layer
    end
    
    methods
        function obj = NeuralNetwork(X, k, X_r, X_c)
            % Constructor to initialize the weights
            obj.X = X;
            obj.k = k;
            obj.X_r = X_r;
            obj.X_c = X_c;
            obj.W1 = randn(X_c + 1, k);
        end
        
        function A = addBias(obj, A)
            % Add a column of ones to the input matrix X
            onesColumn = ones(obj.X_r, 1);
            A = [A onesColumn];
        end
        
        function obj = firstLayer(obj, activation_function)
            % Perform the forward pass and apply the activation function
            obj.X = obj.addBias(obj.X);
            Z = obj.X * obj.W1;
            obj.U = activation_function(Z);
        end

        function obj = secondLayer(obj, m)
                
             obj.U = obj.addBias(obj.U);
             obj.W2 = randn(obj.k+1, m);
        end
    end
end

