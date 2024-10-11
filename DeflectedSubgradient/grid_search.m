function [results, W1, W2, W1_train, W2_train] = grid_search(X, Y, val_X, val_Y, params, plot_results)
    % Performs grid search for the Deflected Subgradient method in a neural network.
    % Explores multiple combinations of hyperparameters: activation function, k, delta, rho, R, lambda, and max iterations.
    % 
    % INPUT:
    %   X            - Training input data.
    %   Y            - Training output data.
    %   val_X        - Validation input data.
    %   val_Y        - Validation output data.
    %   params       - Grid search parameters: activation_functions, k_values, delta_values, rho_values, R_values, lambda_values, max_iter.
    %   plot_results - Boolean flag to enable/disable plotting.
    %
    % OUTPUT:
    %   results  - Results of the grid search (activation, k, delta, rho, R, lambda, max_iter, time, train eval, validation eval, status).
    %   W1       - Optimal first layer weights for the best validation result.
    %   W2       - Optimal second layer weights for the best validation result.
    %   W1_train - Optimal first layer weights for the best training result.
    %   W2_train - Optimal second layer weights for the best training result.


    % Initialize cell array to store results
    num_combinations = numel(params.activation_functions) * numel(params.k_values) * ...
                        numel(params.lambda_values) * numel(params.rho_values) * ...
                        numel(params.R_values) * numel(params.delta_values) * ...
                        numel(params.max_iter);
    results = cell(num_combinations, 10);
    index = 1;

    % Initialize a temporary variable to track the best evaluation score
    temp = inf;
    % Initialize a temporary variable to track the best training score
    temp_train=inf;

    % Iterate over all parameter combinations
    for i = 1:numel(params.activation_functions)
        for k = params.k_values
            for delta = params.delta_values
                for rho = params.rho_values
                    for R = params.R_values
                        for lambda = params.lambda_values
                            for max_iter = params.max_iter

                                % Extract the current activation function and its name
                                activation_function = params.activation_functions{i};
                                activation_function_name = params.activation_functions_names{i};
                                
                                %% Training step
                                
                                % Initialize the neural network for the training data
                                nn = NeuralNetwork(X, k, size(X, 1), size(X, 2));
                                nn = nn.firstLayer(activation_function);
                                nn = nn.secondLayer(size(Y,2));
                                
                                % Initialize the DeflectedSubgradient object
                                ds = DeflectedSubgradient(X, Y, nn.W2, delta, rho, R, ...
                                                  max_iter, nn.U, Y, lambda, plot_results);
                                % Compute the minum with DeflectedSubgradient
                                [x_opt, ds, status] = ds.compute_deflected_subgradient();
                                eval = ds.evaluate_result(x_opt);

                                %% Validation step

                                % Initialize the neural network with learned W1 and x_opt
                                val_nn = NeuralNetwork(val_X, k, size(val_X, 1), size(X,2), nn.W1, x_opt);
                                val_nn = val_nn.firstLayer(activation_function);
                                val_nn = val_nn.secondLayer(size(val_Y, 2));
                                % Evaluate validation set
                                validation_evaluation = val_nn.evaluateModel(val_Y, val_nn.W2);

                                %% Store results 

                                results{index, 1}  = activation_function_name;
                                results{index, 2}  = k;
                                results{index, 3}  = lambda;
                                results{index, 4}  = rho;
                                results{index, 5}  = R;
                                results{index, 6}  = delta;
                                results{index, 7}  = max_iter;
                                results{index, 8} = status;
                                results{index, 9}  = ds.elapsed_time;
                                results{index, 10}  = eval;
                                results{index, 11} = validation_evaluation;
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
            end
        end
    end
end