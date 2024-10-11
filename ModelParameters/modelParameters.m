function params=modelParameters()
    
    addpath("activation_functions")
    
    % Initialize params as a struct
    params = struct();
    
    % Assign values to the fields of params for neural network
    params.activation_functions = {@relu, @tanh, @sigmoid};
    params.activation_functions_names = {'relu', 'tanh', 'sigmoid'};
    params.k_values = [50];
    params.lambda_values = [5e-4];

  