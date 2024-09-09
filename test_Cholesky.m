function test_results=test_Cholesky(X, Y, X_r, X_c, W1, W2, activation_function_str, layer_dim, lambda)
    
    
    % Convert the activation function string to a function handle
    activation_function = str2func(activation_function_str);
    %test_results=cell(3, 4);

    onesColum=ones(X_r, 1);
    X = [X onesColum];
    Z = X * W1;
    U = activation_function(Z);
    U = [U onesColum];
    D = U * W2;

    residual = round(U * W2) - Y;
    frob_norm_squared = sum(sum(residual.^2));
    objective_value = (1 / (2 * X_r)) * frob_norm_squared;
    %fprintf('Objective function value: %f\n', objective_value);

    test_results = objective_value;

    display(test_results);
    

end