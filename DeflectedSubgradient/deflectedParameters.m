function params=deflectedParameters()
    
    % Initialize params as a struct
    params = struct();
    
    % Assign values to the fields of params for neural network
    % params.delta_values = [0.01, 0.001];
    % params.rho_values = [0.15, 0.2, 0.3];
    % params.R_values = [200, 250, 320];
    % params.max_iter = [150, 250, 300];

    % Assign values to the fields of params for neural network
    params.delta_values = [0.01];
    params.rho_values = [0.2];
    params.R_values = [500];
    params.max_iter = [10000];