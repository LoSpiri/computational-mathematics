% Filename: NeuralNetwork.m
classdef NeuralNetwork
    properties
        W1  % Weights for the first layer
    end
    
    methods
        function obj = NeuralNetwork(X_c, W1_c)
            % Constructor to initialize the weights
            obj.W1 = randn(X_c + 1, W1_c);
        end
        
        function X = addBias(~, X)
            % Add a column of ones to the input matrix X
            [X_r, ~] = size(X);
            onesColumn = ones(X_r, 1);
            X = [X onesColumn];
        end
        
        function U = activate(obj, X, activation_function)
            % Perform the forward pass and apply the activation function
            X = obj.addBias(X);
            Z = X * obj.W1;
            U = activation_function(Z);
        end
    end
end

