function [results, W1, W2] = grid_search(X, Y, X_r, X_c, val_X, val_Y, val_X_r, params, plot_results)
    % Initialize cell array to store results
    num_combinations = numel(params.activation_functions) * numel(params.k_values) * ...
                        numel(params.delta_values) * numel(params.rho_values) * ...
                        numel(params.R_values) * numel(params.lambda_values) * ...
                        numel(params.max_iter);
    results = cell(num_combinations, 10);
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

                                % Initialize Neural Network
                                nn = NeuralNetwork(X, k, X_r, X_c);
                                nn = nn.firstLayer(activation_function);
                                nn = nn.secondLayer(size(Y, 2));

                                % Initialize W2 and other parameters
                                W2 = rand(size(nn.W2)); % Adjust size accordingly
                                learning_rate = 0.01; % Set learning rate

                                for iter = 1:max_iter
                                    % Compute the gradients for W2
                                    [grad_W2] = compute_gradients(nn, X, Y, lambda);

                                    % Update W2 using gradient descent
                                    W2 = W2 - learning_rate * grad_W2;

                                    % Optional: Add convergence check
                                    if norm(grad_W2) < 1e-6
                                        break; % Stop if gradients are small
                                    end
                                end

                                % Update nn with optimized W2
                                nn.W2 = W2;

                                % Evaluate the result using the updated W2
                                eval = nn.evaluateModel(Y, nn.W2); % Evaluate using updated W2

                                %% Validation step
                                % Initialize the neural network with learned W1 and W2
                                val_nn = NeuralNetwork(val_X, k, val_X_r, X_c, nn.W1, W2);
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
                                results{index, 8}  = nan; % Placeholder for elapsed_time
                                results{index, 9}  = eval;
                                results{index, 10} = validation_evaluation;
                                index = index + 1;

                                % Save the best configuration based on validation result
                                if validation_evaluation < temp
                                    temp = validation_evaluation;
                                    W1 = nn.W1;
                                    W2 = W2; % Store the optimized W2
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function [grad_W2] = compute_gradients(nn, X, Y, lambda)
    % Compute gradients for W2 based on the current model
    % This function needs to be implemented based on your network's structure
    % and how the loss is calculated.
    
    % Example:
    m = size(X, 1); % Number of samples
    predictions = nn.U * nn.W2; % Forward pass to get predictions
    error = predictions - Y; % Compute error
    
    % Compute gradient (example using L2 loss + L1 regularization)
    grad_W2 = (1/m) * (nn.U' * error) + lambda * sign(nn.W2); % Gradient calculation
end
