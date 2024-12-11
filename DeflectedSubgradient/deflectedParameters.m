function params=deflectedParameters()
    
    % Initialize params as a struct
    params = struct();
    
    % Assign values to the fields of params for neural network
    params.delta_values = [0.5];
    params.rho_values = [0.95];
    params.R_values = [1];
    params.max_iter = [50000];

    % Assign values to the fields of params for neural network
    % params.delta_values = [0.001];
    % params.rho_values = [0.3];
    % params.R_values = [300];
    % params.max_iter = [20000];