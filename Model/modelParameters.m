function params=modelParameters()
    % Initializes and returns `params`, a struct with neural network parameters.
    
    % OUTPUT:
    %   params - Struct containing:
    %     - activation_functions: Cell array of function handles.
    %     - activation_functions_names: Cell array of function names.
    %     - k_values: List of neuron counts for hidden layers.
    %     - lambda_values: List of regularization values.

    
    addpath("Model/activation_functions")
    
    % Initialize params as a struct
    params = struct();
    
    % Assign values to the fields of params for neural network
    params.activation_functions = {@relu, @tanh, @sigmoid};
    params.activation_functions_names = {'relu', 'tanh', 'sigmoid'};
    params.k_values = [18, 22, 26, 30];
    params.lambda_values = [5e-4, 1e-4, 1e-5];

  