function [results, W1, W2, W1_train, W2_train] = grid_search(X, Y, val_X, val_Y, ...
                                                modelParams, deflectedParams, plots)
    % Performs a grid search for hyperparameter optimization
    % of the Deflected Subgradient method in a neural network. It explores 
    % different combinations of activation functions, k, lambda, and other 
    % hyperparameters to find the best model for both training and validation sets.
    %
    % INPUT:
    %   X               - Training input data matrix.
    %   Y               - Training output data matrix.
    %   val_X           - Validation input data matrix.
    %   val_Y           - Validation output data matrix.
    %   modelParams     - Struct containing the grid search parameters, including:
    %                   - activation_functions
    %                   - activation_functions_names
    %                   - k_values
    %                   - lambda_values
    %   deflectedParams - Parameters for the Deflected Subgradient method:
    %                   - rho_values, R_values, delta_values, max_iter
    %   plots           - Boolean flag to enable/disable plotting during training.
    %
    % OUTPUT:
    %   results  - A cell array with the grid search results for each combination
    %              of hyperparameters, including validation results and training results.
    %   W1       - Best first-layer weights for the optimal validation result.
    %   W2       - Best second-layer weights for the optimal validation result.
    %   W1_train - Best first-layer weights based on training performance.
    %   W2_train - Best second-layer weights based on training performance.

    % Initialize the results cell array to store each combination's results
    num_combinations = numel(modelParams.activation_functions) * ...
                        numel(modelParams.k_values) * numel(modelParams.lambda_values) * 2;  % 2: for validation and training results
    results = cell(num_combinations, 11);  

    index = 1;  

    
    for i = 1:numel(modelParams.activation_functions)
                          
        % Extract the current activation function and its name                               
        act_fun = modelParams.activation_functions{i};                              
        act_fun_name = modelParams.activation_functions_names{i};  

        for k = modelParams.k_values
            for lambda = modelParams.lambda_values

                % Call the function to train and evaluate the model with current parameters
                [results, W1, W2, W1_train, W2_train] = bestDeflected(deflectedParams, ...
                                                 X, Y, val_X, val_Y, k, lambda, act_fun, ...
                                                 act_fun_name, index, results, plots);
                index = index + 2;  
            end
        end
    end
end



function [results, W1, W2, W1_train, W2_train] = bestDeflected(params, X, Y, val_X, val_Y, ...
                                                  k, lambda, act_fun, act_fun_name, index, ...
                                                  results, plots)
    % Trains a neural network using Deflected Subgradient 
    % method and evaluates its performance on both training and validation sets.
    %
    % INPUT:
    %   params       - Deflected Subgradient parameters (rho_values, R_values, delta_values, max_iter).
    %   X            - Training input data.
    %   Y            - Training output data.
    %   val_X        - Validation input data.
    %   val_Y        - Validation output data.
    %   k            - Number of neurons in the first hidden layer.
    %   lambda       - Regularization parameter.
    %   act_fun      - Activation function used for the neural network.
    %   act_fun_name - Name of the activation function (for storing in results).
    %   index        - Index to store the current combination results in the results array.
    %   results      - Cell array to store results.
    %   plots        - Boolean flag to enable/disable plotting during training.
    %
    % OUTPUT:
    %   results    - Updated results cell array with training and validation evaluations.
    %   W1         - Best first-layer weights for the validation set.
    %   W2         - Best second-layer weights for the validation set.
    %   W1_train   - Best first-layer weights based on training set evaluation.
    %   W2_train   - Best second-layer weights based on training set evaluation.

    % Initialize temporary variables for validation and training best scores
    temp = inf;        
    temp_train = inf;  

    for rho = params.rho_values
        for R = params.R_values
            for delta = params.delta_values
                for max_iter = params.max_iter
                
                %% Training step

                % Initialize the neural network for training
                nn = NeuralNetwork(X, k, size(X, 1), size(X, 2)); 
                nn = nn.firstLayer(act_fun);  
                nn = nn.secondLayer(size(Y,2));  
                                
                % Initialize the Deflected Subgradient object with training parameters
                ds = DeflectedSubgradient(X, Y, nn.W2, delta, rho, R, ...
                                        max_iter, nn.U, Y, lambda, plots);
                % Optimize the network using Deflected Subgradient method
                [x_opt, ds, status] = ds.compute_deflected_subgradient();
                eval = ds.evaluate_result(x_opt);  

                %% Validation step

                % Initialize the neural network with trained W1 and the optimized W2
                val_nn = NeuralNetwork(val_X, k, size(val_X, 1), size(X, 2), nn.W1, x_opt);
                val_nn = val_nn.firstLayer(act_fun);  
                val_nn = val_nn.secondLayer(size(val_Y, 2));  
                % Evaluate the model on the validation set
                validation_evaluation = val_nn.evaluateModel(val_Y, val_nn.W2);

                %% Store results 

                % Save the best configuration based on validation result
                if validation_evaluation < temp
                    temp = validation_evaluation;  
                    W1 = nn.W1;  
                    W2 = x_opt; 

                    % Store the results for the best validation result
                    results{index, 1}  = act_fun_name;  
                    results{index, 2}  = k;  
                    results{index, 3}  = lambda;  
                    results{index, 4}  = rho;  
                    results{index, 5}  = R;  
                    results{index, 6}  = delta; 
                    results{index, 7}  = max_iter;  
                    results{index, 8}  = status;  
                    results{index, 9}  = ds.elapsed_time;  
                    results{index, 10} = eval;  
                    results{index, 11} = validation_evaluation; 
                end

                % Save the best configuration based on train result
                if eval < temp_train
                    temp_train = eval;  
                    W1_train = nn.W1; 
                    W2_train = x_opt;  

                    % Store the results for the training result
                    results{index+1, 1}  = act_fun_name;  
                    results{index+1, 2}  = k;  
                    results{index+1, 3}  = lambda;  
                    results{index+1, 4}  = rho;  
                    results{index+1, 5}  = R;  
                    results{index+1, 6}  = delta;  
                    results{index+1, 7}  = max_iter;  
                    results{index+1, 8}  = status;  
                    results{index+1, 9}  = ds.elapsed_time;  
                    results{index+1, 10} = eval;  
                    results{index+1, 11} = validation_evaluation; 
                end

                end                        
            end
        end
    end
end
