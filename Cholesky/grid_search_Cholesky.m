function [results, W1, W2, W1_train, W2_train] = grid_search_Cholesky(X, Y, val_X, val_Y, params)
    % Performs grid search for Cholesky-based least squares in a neural network.
    % Tests various activation functions, K values, and regularization (lambda).
    % 
    % INPUT:
    %   X       - Training input data.
    %   Y        - Training output data.
    %   val_X    - Validation input data.
    %   val_Y    - Validation output data.
    %   params   - Grid search parameters: activation_functions, k_values, lambda_values.
    % 
    % OUTPUT:
    %   results  - Results of the grid search (activation, K, lambda, time, train and validation evals).
    %   W1       - Optimal first layer weights for the best validation result.
    %   W2       - Optimal second layer weights for the best validation result.
    %   W1_train - Optimal first layer weights for the best training result.
    %   W2_train - Optimal second layer weights for the best training result.

    % Initialize cell array to store results: 6 columns
    num_combinations = numel(params.activation_functions) * ...
                       numel(params.k_values) * numel(params.lambda_values);
    results = cell(num_combinations, 6); 
    index = 1;

    % Initialize a temporary variable to track the best evaluation score
    temp = inf;
    % Initialize a temporary variable to track the best training score
    temp_train=inf;

    % Iterate over all parameter combinations
    for i = 1:numel(params.activation_functions)
        for k = params.k_values
            for lambda = params.lambda_values

                % Extract the current activation function and its name
                activation_function = params.activation_functions{i};
                activation_function_name = params.activation_functions_names{i};

                %% Training step
                
                % Initialize the neural network for the training data
                nn = NeuralNetwork(X, k, size(X,1), size(X,2));
                nn = nn.firstLayer(activation_function);
                nn = nn.secondLayer(size(Y,2));
                
                % Initialize the Cholesky least squares problem
                chol = CholeskyLeastSquares(nn.U, Y, lambda);
                % Compute the Cholesky decomposition
                chol = chol.computeCholesky();
                % Solve the least squares problem
                [x_opt, chol] = chol.solve();
                elapsed_time = chol.ComputeCholeskyTime;
                eval = chol.evaluateResult(x_opt);

                %% Validation step

                % Initialize the neural network with learned W1 and x_opt
                val_nn = NeuralNetwork(val_X, k, size(val_X,1), size(X, 2), nn.W1, x_opt);
                val_nn = val_nn.firstLayer(activation_function);
                val_nn = val_nn.secondLayer(size(val_Y, 2));
                % Evaluate validation set
                validation_evaluation = val_nn.evaluateModel(val_Y, x_opt);
                

                %% Store results
                
                results{index, 1} = activation_function_name;
                results{index, 2} = k;
                results{index, 3} = lambda;
                results{index, 4} = elapsed_time;
                results{index, 5} = eval;
                results{index, 6} = validation_evaluation;
                index = index + 1;

                % Save the best configuration based on validation result
                if validation_evaluation < temp
                    temp = validation_evaluation;
                    W1 = nn.W1;
                    W2 = x_opt;
                end

                % Save the best configuration based on train result
                if eval < temp_train
                    temp_train = eval;
                    W1_train = nn.W1;
                    W2_train = x_opt;
                end

            end
        end
    end
end