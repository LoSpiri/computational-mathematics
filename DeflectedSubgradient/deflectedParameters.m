function params = deflectedParameters()
    % DEFLECTEDPARAMETERS Returns a struct containing parameter values for 
    % the deflected subgradient method.
    %
    % This function initializes and returns a struct `params` with 
    % predefined parameter values used in the deflected subgradient method 
    % for optimization or training.
    %
    % OUTPUT:
    %   - params: A struct with the following fields:
    %       * delta_values: A variable indicating the required improvement 
    %         in the objective function.
    %       * rho_values: A constant specifying the adjustment factor for delta.
    %       * R_values: A threshold for updating delta.
    %       * max_iter: The maximum number of iterations.
    %       * min_alpha: The minimum allowable value for alpha.


    % Initialize params as a struct
    params = struct();

    % params.delta_values = [0.8, 0.9, 1, 1.1, 1.2];
    % params.rho_values = [0.90, 0.95, 0.99];
    % params.R_values = [0.45, 0.5, 0.55];
    % params.max_iter = [50000];
    % params.min_alpha = [0.01];

    % params.delta_values = [0.1, 0.5, 1, 5, 10];
    % params.rho_values = [0.1, 0.3, 0.7, 0.95];
    % params.R_values = [0.2, 0.5, 0.8, 1, 5];
    % params.max_iter = [50000];
    params.delta_values = [2];
    params.rho_values = [0.99];
    params.R_values = [10];
    params.max_iter = [50000];
    params.min_alpha = [0.01];
    
    % params.delta_values = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10];
    % params.rho_values = [0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99];
    % params.R_values = [0.1, 0.3, 0.4, 0.5, 0.8, 1, 5, 10];
    % params.max_iter = [50000];
    %params.min_alpha = [0.01, 0.001, 0.0001];
end


