function results = grid_search_Cholesky(X, Y, X_r, X_c, params)
    % Initialize cell array to store results
    num_combinations = numel(params.activation_functions) * numel(params.k_values) * numel(params.lambda_values);
    results = cell(num_combinations, 5); % 5 columns: ActivationFunction, KValue, Lambda, ElapsedTime, Evaluation
    index = 1;

    % Iterate over all parameter combinations
    for i = 1:numel(params.activation_functions)
        for k = params.k_values
            for lambda = params.lambda_values
                           
                activation_function = params.activation_functions{i};
                activation_function_name = params.activation_functions_names{i};

                nn = NeuralNetwork(X, k, X_r, X_c);
                nn = nn.firstLayer(activation_function);
                nn = nn.secondLayer(size(Y,2));

                chol = CholeskyLeastSquares(nn.U, Y, lambda);
                chol = chol.computeCholesky();
                [x_opt, chol] = chol.solve();
                elapsed_time = chol.ComputeCholeskyTime+chol.SolveTime;
                eval = chol.evaluateResult(x_opt);

                % Store results in cell array
                results{index, 1} = activation_function_name;
                results{index, 2} = k;
                results{index, 3} = lambda;
                results{index, 4} = elapsed_time;
                results{index, 5} = eval;
                index = index + 1;
                       
            end
        end
    end
end