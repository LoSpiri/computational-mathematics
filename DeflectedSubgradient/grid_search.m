function [results, W1, W2, W1_train, W2_train] = grid_search(X, Y, val_X, val_Y, ...
                                                modelParams, deflectedParams)
    % Performs a grid search for hyperparameter optimization
    % of the Deflected Subgradient method in a neural network. It explores 
    % various combinations of activation functions, number of neurons, regularization
    % parameters, and other hyperparameters to identify the best model based on
    % both training and validation performance.
    %
    % INPUT:
    %   X               - Training input data matrix.
    %   Y               - Training output data matrix.
    %   val_X           - Validation input data matrix.
    %   val_Y           - Validation output data matrix.
    %   modelParams     - Struct containing the grid search parameters, including:
    %                     - activation_functions: Cell array of activation functions.
    %                     - activation_functions_names: Corresponding names of the functions.
    %                     - k_values: Array of neuron counts for hidden layers.
    %                     - lambda_values: Array of regularization parameters.
    %   deflectedParams - Struct with Deflected Subgradient method parameters:
    %                     - rho_values: Array of step size parameters.
    %                     - R_values: Array of threshold values.
    %                     - delta_values: Array of delta parameters for updates.
    %                     - max_iter: Array of maximum iteration values.
    %                     - min_alpha: Array of minimum alpha values.
    %
    % OUTPUT:
    %   results  - A cell array containing results for each hyperparameter combination.
    %              Includes both training and validation results.
    %   W1       - Best first-layer weights for the optimal validation performance.
    %   W2       - Best second-layer weights for the optimal validation performance.
    %   W1_train - Best first-layer weights based on training performance.
    %   W2_train - Best second-layer weights based on training performance.
    

    % Initialize the results cell array to store each combination's results
    num_combinations = numel(modelParams.activation_functions) * ...
                        numel(modelParams.k_values) * ...
                        numel(modelParams.lambda_values) * 2;  
    results = cell(num_combinations, 14);  

    index = 1;  % Index for storing results
    num_iteration=1; % Counter for tracking iterations

    for i = 1:numel(modelParams.activation_functions)

        % Extract the current activation function and its name
        act_fun = modelParams.activation_functions{i};
        act_fun_name = modelParams.activation_functions_names{i};

        for k = modelParams.k_values
            for lambda = modelParams.lambda_values

                % Initialize the neural network for training
                nn = NeuralNetwork(X, k, size(X, 1), size(X, 2)); 
                nn = nn.firstLayer(act_fun);  
                nn = nn.secondLayer(size(Y,2));  

                [results, W1, W2, W1_train, W2_train] = bestDeflected(deflectedParams, ...
                                                 X, Y, val_X, val_Y, k, lambda, act_fun, ...
                                                 act_fun_name, index, results, num_iteration, ...
                                                 nn.U, nn.W2, nn.W1);
                
                % Update indices and iteration count
                index = index + 2;
                num_iteration = num_iteration + ...
                                (numel(deflectedParams.rho_values) * ...
                                numel(deflectedParams.max_iter) * ...
                                numel(deflectedParams.R_values) * ...
                                numel(deflectedParams.delta_values) * ...
                                numel(deflectedParams.min_alpha));
            end
        end
    end
end



function [results, W1, W2, W1_train, W2_train] = bestDeflected(params, X, Y, val_X, val_Y, ...
                                                  k, lambda, act_fun, act_fun_name, index, ...
                                                  results, num_iteration, U_iniz, W2_iniz, W1_iniz)
    % Trains a neural network using Deflected Subgradient 
    % method and evaluates its performance on both training and validation sets.
    %
    % INPUT:
    %   params       - Deflected Subgradient parameters.
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
    %   num_iteration- Current iteration count for logging.
    %
    % OUTPUT:
    %   results    - Updated results cell array with training and validation evaluations.
    %   W1         - Best first-layer weights for the validation set.
    %   W2         - Best second-layer weights for the validation set.
    %   W1_train   - Best first-layer weights based on training set evaluation.
    %   W2_train   - Best second-layer weights based on training set evaluation.


    % Initialize variables to track the best results
    temp = inf;        
    temp_train = inf;  

    for rho = params.rho_values
        for R = params.R_values
            for delta = params.delta_values
                for max_iter = params.max_iter
                    for min_alpha = params.min_alpha
                
                    %% Training step
                    
                     % Initialize the neural network for training
                     nn = NeuralNetwork(X, k, size(X, 1), size(X, 2), W1_iniz, W2_iniz); 
                     nn = nn.firstLayer(act_fun);  
                     nn = nn.secondLayer(size(Y,2)); 

                    % Initialize the Deflected Subgradient object with current parameters
                    ds = DeflectedSubgradient(U_iniz, Y, W2_iniz, delta, rho, R, ...
                                            max_iter, min_alpha, lambda);
                    % Optimize the network using Deflected Subgradient method
                    [x_opt, values_arrays, ds, status] = ds.compute_deflected_subgradient();
                    eval = ds.evaluate_f(x_opt); % Evaluate training performance
    
                    %% Validation step
    
                    % Initialize the neural network with trained W1 and the optimized W2
                    val_nn = NeuralNetwork(val_X, k, size(val_X, 1), size(X, 2), W1_iniz, x_opt);
                    val_nn = val_nn.firstLayer(act_fun);  
                    val_nn = val_nn.secondLayer(size(val_Y, 2));

                    % Evaluate the model on the validation set
                    validation_evaluation = val_nn.evaluateModel(val_Y, val_nn.W2);
    
                    %% Store results 
    
                    % Update best configuration based on validation performance
                    if validation_evaluation < temp
                        temp = validation_evaluation;  
                        W1 = val_nn.W1;  
                        W2 = x_opt; 
    
                        % Store validation results
                        results{index, 1}  = act_fun_name;  
                        results{index, 2}  = k;  
                        results{index, 3}  = lambda;  
                        results{index, 4}  = rho;  
                        results{index, 5}  = R;  
                        results{index, 6}  = delta; 
                        results{index, 7}  = max_iter;
                        results{index, 8}  = min_alpha;
                        results{index, 9}  = status;  
                        results{index, 10}  = ds.elapsed_time;  
                        results{index, 11} = eval;  
                        results{index, 12} = validation_evaluation;
                        results{index, 13} = values_arrays;
                        results{index, 14} = "nn";
                    end
    
                    % Update best configuration based on training performance
                    if eval < temp_train
                        temp_train = eval;  
                        W1_train = W1; 
                        W2_train = x_opt;  
    
                        % Store training results
                        results{index+1, 1}  = act_fun_name;  
                        results{index+1, 2}  = k;  
                        results{index+1, 3}  = lambda;  
                        results{index+1, 4}  = rho;  
                        results{index+1, 5}  = R;  
                        results{index+1, 6}  = delta;  
                        results{index+1, 7}  = max_iter;
                        results{index+1, 8}  = min_alpha;
                        results{index+1, 9}  = status;  
                        results{index+1, 10}  = ds.elapsed_time;  
                        results{index+1, 11} = eval;  
                        results{index+1, 12} = validation_evaluation;
                        results{index+1, 13} = values_arrays;
                        results{index+1, 14} = "method";
                    end
                    
                    % Print the number of the current iteration
                    fprintf('Iteration number %d\n', num_iteration);
                    num_iteration=num_iteration+1;
                    end
                end                        
            end
        end
    end
end
