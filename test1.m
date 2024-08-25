function test()
    % Minimal test case
    activation_functions = {@tanh};  % Ensure this is a cell array
    activation_functions_names = {'tanh'};
    
    params = struct(...
        'activation_functions', activation_functions, ...
        'activation_functions_names', activation_functions_names, ...
        'k_values', [5], ...
        'delta_values', [0.1], ...
        'rho_values', [0.1], ...
        'R_values', [1], ...
        'lambda_values', [0.01]);
    
    % Display type and contents
    disp('Before calling grid_search:');
    disp('Type of params.activation_functions:');
    disp(class(params.activation_functions));
    disp('Contents of params.activation_functions:');
    disp(params.activation_functions);
    
    results = grid_search([], [], 1, params);
    disp(results);
end

function results = grid_search(X, Y, X_r, params)
    disp('Inside grid_search:');
    disp('Type of params.activation_functions:');
    disp(class(params.activation_functions));
    disp('Contents of params.activation_functions:');
    disp(params.activation_functions);
    
    for i = 1:numel(params.activation_functions)
        activation_function = params.activation_functions{i};  % Should be valid if it’s a cell array
        activation_function_name = params.activation_functions_names{i};  % Should be valid if it’s a cell array
        disp(activation_function_name); % Just to verify the loop is correct
    end
    results = {};
end

test();