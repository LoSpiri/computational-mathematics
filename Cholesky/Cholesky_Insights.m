%% Main Function
function Cholesky_Insights(best_result, W1, X, Y)
    
    % Print best results
    results_table = cell2table(best_result, 'VariableNames', {'ActivationFunction', ...
                               'KValue', 'Lambda', 'ElapsedTime', 'Evaluation'});

    fprintf('Best Configuration for Cholesky Method:\n');
    disp(results_table);
    
    activation_str=best_result{1,1};
    k = best_result{1,2};  % Esempio: estrai il valore k migliore
    lambda = best_result{1,3};  % Esempio: estrai il valore lambda migliore

    % Convert the activation function string to a function handle
    activation_function = str2func(activation_str);

    % Initialize the neural network for the training data
    nn = NeuralNetwork(X, k, size(X,1), size(X,2), W1);
    nn = nn.firstLayer(activation_function);
    nn = nn.secondLayer(size(Y,2));

    % Initialize and solve the Cholesky least squares problem
    chol = CholeskyLeastSquares(nn.U, Y, lambda);
    % Compute the Cholesky decomposition
    chol = chol.computeCholesky();
    % Solve the least squares problem
    [x_opt, chol] = chol.solve();
    elapsed_time = chol.ComputeCholeskyTime;
    eval = chol.evaluateResult(x_opt);

    % Step 1: Calcolo del numero di condizionamento di A^T A
    cond_number = cond(chol.AtA);  % Numero di condizionamento di A^T A
    fprintf('Numero di condizionamento di A^T A: %.5e\n', cond_number);


    % Step 2: Calcolo del numero di condizionamento relativo
    cos_theta = norm(chol.A*x_opt)/norm(chol.B);
    k_rel = cond_number/cos_theta;
    fprintf('Numero che esprime cos_theta: %.4f\n', cos_theta);
    fprintf('Numero di condizionamento relativo kappa_rel: %.4f\n', k_rel);

    % Step 3: Calcolo dell'errore relativo nella decomposizione
    RtR = chol.R'*chol.R;
    rel_error_decomp = norm(chol.AtA - RtR) / norm(chol.AtA);
    fprintf('Errore relativo nella decomposizione: %.5e\n', rel_error_decomp);

    % Step 4: Calcolo del residuo relativo
    residual = norm(chol.A * x_opt - chol.B) / norm(chol.B);
    fprintf('Residuo relativo: %.5e\n', residual);

end




