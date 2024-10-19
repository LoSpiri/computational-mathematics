function params=deflectedParameters()
    
    % Initialize params as a struct
    params = struct();
    
    % Assign values to the fields of params for neural network
    params.delta_values = [0.001];
    params.rho_values = [0.2];
    params.R_values = [256];
    params.max_iter = [250];

    % Assign values to the fields of params
    % params.activation_functions = {@relu};
    % params.activation_functions_names = {'relu'};
    % params.k_values = [2, 4, 8, 16, 32, 64, 128, 256];
    % params.delta_values = [0.001, 0.01, 0.25, 0.5];
    % params.rho_values = [0.1, 0.25, 0.5, 0.75, 0.95];
    % params.R_values = [2, 4, 8, 16, 32, 64, 128, 256];
    % params.lambda_values = [1e-4, 3e-3, 4e-5];
    % params.max_iter = [100, 250];
    
    % params.activation_functions = {@relu};
    % params.activation_functions_names = {'relu'};
    % params.k_values = [2, 4, 8, 16, 32];
    % params.delta_values = [0.001, 0.01];
    % params.rho_values = [0.1, 0.25, 0.5];
    % params.R_values = [2, 8, 32];
    % params.lambda_values = [1e-4, 3e-3];
    % params.max_iter = [100, 200, 1000];