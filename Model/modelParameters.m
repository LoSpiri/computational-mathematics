function params = modelParameters()
    % MODELPARAMETERS Returns a struct containing neural network parameters.
    %
    % This function initializes and returns a struct `params` with 
    % predefined parameters commonly used for configuring neural networks.
    %
    % OUTPUT:
    %   - params: A struct with the following fields:
    %       * activation_functions: A cell array of function handles representing 
    %         different activation functions.
    %       * activation_functions_names: A cell array of strings representing 
    %         the names of the activation functions.
    %       * k_values: A list of neuron counts for the hidden layers.
    %       * lambda_values: A list of regularization parameter values.


    
    addpath("Model/activation_functions")
    
    % Initialize params as a struct
    params = struct();

    % Assign values to the fields of params for neural network
    params.activation_functions = {@sigmoid};
    params.activation_functions_names = {'sigmoid'};
    params.k_values = [800];
    params.lambda_values = [1e-3];
 
end