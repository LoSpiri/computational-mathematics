function params=deflectedParameters()
    
    % Initialize params as a struct
    params = struct();
    
    % Assign values to the fields of params for neural network
    params.delta_values = [0.01, 0.001];
    params.rho_values = [0.2, 0.3];
    params.R_values = [150, 200, 300];
    params.max_iter = [15000];

    % Assign values to the fields of params for neural network
    % params.delta_values = [0.01];
    % params.rho_values = [0.2];
    % params.R_values = [500];
    % params.max_iter = [10000];