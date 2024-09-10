function [results, W1, W2] = grid_search_Cholesky(X, Y, X_r, X_c, val_X, val_Y, val_X_r, params)
    % Performs grid search with Cholesky decomposition.
    %
    % INPUT:
    %   X      - Input data matrix.
    %   Y      - Output data matrix.
    %   X_r    - Number of rows in X.
    %   X_c    - Number of columns in X.
    %   params - Struct containing grid search parameters:
    %            activation_functions, k_values, lambda_values.
    %
    % OUTPUT:
    %   results - Cell array with results of the grid search.
    %   W1      - Weights of the first layer for the best configuration.
    %   W2      - Weights of the second layer for the best configuration.

    % Initialize cell array to store results
    num_combinations = numel(params.activation_functions) * numel(params.k_values) * numel(params.lambda_values);
    % 6 columns: ActivationFunction, KValue, Lambda, ElapsedTime,
    % Train_Evaluation, Validation_Evaluation
    results = cell(num_combinations, 6); 
    index = 1;

    % Initialize a temporary variable to track the best evaluation score
    temp = 100;

    % Iterate over all parameter combinations
    for i = 1:numel(params.activation_functions)
        for k = params.k_values
            for lambda = params.lambda_values
                
                % Extract the current activation function and its name
                activation_function = params.activation_functions{i};
                activation_function_name = params.activation_functions_names{i};
                
                % Initialize the neural network with the given parameters
                nn = NeuralNetwork(X, k, X_r, X_c);
                nn = nn.firstLayer(activation_function);
                nn = nn.secondLayer(size(Y,2));
                
                % Initialize Cholesky least squares solver
                chol = CholeskyLeastSquares(nn.U, Y, lambda);
                % Compute the Cholesky decomposition
                chol = chol.computeCholesky();
                % Solve the least squares problem
                [x_opt, chol] = chol.solve();
                elapsed_time = chol.ComputeCholeskyTime+chol.SolveTime;
                eval = chol.evaluateResult(x_opt);

                %Test on evaluation
                onesColum=ones(val_X_r, 1);
                Z = [val_X onesColum];
                Z = Z * nn.W1;
                U = activation_function(Z);
                U = [U onesColum];
                residual = round(U * x_opt) - val_Y;
                frob_norm_squared = sum(sum(residual.^2));
                validation_evaluation = (1 / (val_X_r)) * frob_norm_squared;

                % Store results in cell array
                results{index, 1} = activation_function_name;
                results{index, 2} = k;
                results{index, 3} = lambda;
                results{index, 4} = elapsed_time;
                results{index, 5} = eval;
                results{index, 6} = validation_evaluation;
                index = index + 1;

                % Save layers' weights if this is the best result so far
                if validation_evaluation < temp
                    temp = eval;
                    W1 = nn.W1;
                    W2 = nn.W2;
                end         
            end
        end
    end
end