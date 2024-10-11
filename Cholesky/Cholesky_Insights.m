function Cholesky_Insights(best_result, W1, X, Y)
    % This function provides insights into the Cholesky method's performance
    % for the best configuration, printing key metrics like conditioning numbers,
    % relative error, and residuals.
    
    % INPUT:
    %   best_result - A cell array containing the best combination of parameters.
    %   W1          - Weights of the first layer of the neural network.
    %   X, Y        - Input and output data matrices.
    
    % OUTPUT:
    %   None. Results are printed to the console.
    
    % Print best result configuratiion
    results_table = cell2table(best_result, 'VariableNames', {'ActivationFunction', ...
                               'KValue', 'Lambda', 'ElapsedTime', 'Evaluation'});
    fprintf('Best Configuration for Cholesky Method:\n');
    disp(results_table);
    
    % Extract activation function, k value, and lambda 
    activation_str=best_result{1,1};
    k = best_result{1,2};  
    lambda = best_result{1,3};  

    % Convert the activation function string to a function handle
    activation_function = str2func(activation_str);

    %% Training Steps

    % Initialize the neural network with the best W1 weights.
    nn = NeuralNetwork(X, k, size(X,1), size(X,2), W1);
    nn = nn.firstLayer(activation_function);
    nn = nn.secondLayer(size(Y,2));

    % Solve the least squares problem using Cholesky decomposition
    chol = CholeskyLeastSquares(nn.U, Y, lambda);
    chol = chol.computeCholesky();
    [x_opt, chol] = chol.solve();

    %% Method Analysis

    % Step 1: Calculate the condition number of A^T A.
    cond_number = cond(chol.AtA);  % Numero di condizionamento di A^T A
    fprintf('Numero di condizionamento di A^T A: %.5e\n', cond_number);

    % Step 2: Calculate the relative condition number using cos_theta.
    cos_theta = norm(chol.A*x_opt)/norm(chol.B);
    k_rel = cond_number/cos_theta;
    fprintf('Numero che esprime cos_theta: %.4f\n', cos_theta);
    fprintf('Numero di condizionamento relativo kappa_rel: %.4f\n', k_rel);

    % Step 3: Calculate the relative error in the Cholesky decomposition.
    RtR = chol.R'*chol.R;
    rel_error_decomp = norm(chol.AtA - RtR) / norm(chol.AtA);
    fprintf('Errore relativo nella decomposizione: %.5e\n', rel_error_decomp);

    % Step 4: Calculate the relative residual.
    residual = norm(chol.A * x_opt - chol.B) / norm(chol.B);
    fprintf('Residuo relativo: %.5e\n', residual);

end




