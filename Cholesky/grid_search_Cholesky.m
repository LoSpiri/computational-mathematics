function [results, W1, W2, W1_train, W2_train] = grid_search_Cholesky(X, Y, val_X, val_Y, params)
    % Performs a grid search with Cholesky decomposition
    % and evaluates on both training and validation sets.
    %
    % INPUT:
    %   X       - Input data matrix for training.
    %   Y       - Output data matrix for training.
    %   val_X   - Input data matrix for validation.
    %   val_Y   - Output data matrix for validation.
    %   params  - Struct containing grid search parameters with fields:
    %             activation_functions, activation_functions_names, k_values, lambda_values.
    %
    % OUTPUT:
    %   results - A cell array with the results of the grid search. Each row 
    %             corresponds to a combination of parameters, containing:
    %             ActivationFunction, KValue, Lambda, ElapsedTime, 
    %             Train_Evaluation, Validation_Evaluation.
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
    % Initialize a temporary variable to track the best training score
    temp_train=100;

    % Iterate over all parameter combinations
    for i = 1:numel(params.activation_functions)
        for k = params.k_values
            for lambda = params.lambda_values

                % Extract the current activation function and its name
                activation_function = params.activation_functions{i};
                activation_function_name = params.activation_functions_names{i};

                %Training part
                
                % Initialize the neural network for the training data
                nn = NeuralNetwork(X, k, size(X,1), size(X,2));
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

                %% Validation step
                % Initialize the neural network with learned W1 and x_opt
                val_nn = NeuralNetwork(val_X, k, size(val_X,1), size(X, 2), nn.W1, x_opt);
                val_nn = val_nn.firstLayer(activation_function);
                val_nn = val_nn.secondLayer(size(val_Y, 2));
                % Evaluate validation set
                validation_evaluation = val_nn.evaluateModel(val_Y, x_opt);

                % Store results in cell array
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
                    temp = eval;
                    W1_train = nn.W1;
                    W2_train = x_opt;
                end

            end
        end
    end
end