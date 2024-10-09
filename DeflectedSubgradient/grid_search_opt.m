function [results, W1, W2] = grid_search(X, Y, X_r, X_c, val_X, val_Y, val_X_r, params, plot_results)
    % Initialize cell array to store results
    num_combinations = numel(params.activation_functions) * numel(params.k_values) * ...
                        numel(params.delta_values) * numel(params.rho_values) * ...
                        numel(params.R_values) * numel(params.lambda_values) * ...
                        numel(params.max_iter);
    results = cell(num_combinations, 9);
    index = 1;

    % Initialize a temporary variable to track the best evaluation score
    temp = inf;

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

                                nn = NeuralNetwork(X, k, X_r, X_c);
                                nn = nn.firstLayer(activation_function);
                                nn = nn.secondLayer(size(Y,2));

                                % Define the objective function as a function handle
                                objective_function = @(W2) norm(nn.U * W2 - Y, 'fro') / (2 * size(X, 1)) + lambda * norm(W2, 1);

                                % Initial guess for the optimization
                                initial_guess = rand(size(nn.W2)); % Puoi cambiare questo con una migliore inizializzazione

                                % Set optimization options
                                options = optimset('Display', 'iter', 'TolFun', 1e-12, 'TolX', 1e-12, 'MaxIter', max_iter);

                                % Use fminsearch to minimize the objective function
                                tic; % Start timing
                                [x_opt, f_opt] = fminsearch(objective_function, initial_guess, options);
                                elapsed_time = toc; % End timing

                                %% Validation step
                                % Initialize the neural network with learned W1 and x_opt
                                val_nn = NeuralNetwork(val_X, k, val_X_r, X_c, nn.W1, x_opt);
                                val_nn = val_nn.firstLayer(activation_function);
                                val_nn = val_nn.secondLayer(size(val_Y, 2));
                                % Evaluate validation set
                                validation_evaluation = val_nn.evaluateModel(val_Y, val_nn.W2);

                                % Store results in cell array
                                results{index, 1}  = activation_function_name;
                                results{index, 2}  = k;
                                results{index, 3}  = delta;
                                results{index, 4}  = rho;
                                results{index, 5}  = R;
                                results{index, 6}  = lambda;
                                results{index, 7}  = max_iter;
                                results{index, 8}  = elapsed_time;
                                results{index, 9}  = f_opt;
                                results{index, 10} = validation_evaluation;
                                index = index + 1;

                                % Save the best configuration based on validation result
                                if validation_evaluation < temp
                                    temp = validation_evaluation;
                                    W1 = nn.W1;
                                    W2 = x_opt;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
