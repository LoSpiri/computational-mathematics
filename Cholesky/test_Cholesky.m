function test_results = test_Cholesky(results, X, Y, W1, W2)
    % Test the performance of a neural network trained using Cholesky 
    % factorization on a test dataset.
    %
    % INPUT:
    %   results - A cell array containing the best configuration parameters 
    %             from the training phase (activation function, k, lambda)
    %   X       - Input test data matrix
    %   Y       - Output test data matrix
    %   W1      - Learned weight matrix from the first layer
    %   W2      - Learned weight matrix from the second layer
    %
    % OUTPUT:
    %   test_results - A cell array containing the test result of the neural 
    %                  network, including method name, activation function, 
    %                  k value, lambda, and test evaluation result.

    activation_func = results{1, 1};  % Best activation function from training
    k = results{1, 2};                % Best k value from training
    lambda = results{1, 3};           % Best lambda value from training
    
    % Convert the activation function name to a function handle
    activation_function = str2func(activation_func);

    test_results = cell(1, 5);  % Cell array to store the test results

    %% Evaluate the test set and store result
                    
    % Initialize the neural network with the provided weights W1 and W2
    test_nn = NeuralNetwork(X, k, size(X, 1), size(X, 2), W1, W2);
    test_nn = test_nn.firstLayer(activation_function);  % Set activation function in the first layer
    test_nn = test_nn.secondLayer(size(Y, 2));          % Set the size of the second layer (output layer)

    result = test_nn.evaluateModel(Y, test_nn.W2);  % Evaluate the model on the test data using W2
    
    % Store the test result in the cell array
    test_results{1, 1} = 'Cholesky';                % Method name
    test_results{1, 2} = activation_func;           % Activation function used
    test_results{1, 3} = k;                         % K value
    test_results{1, 4} = lambda;                    % Lambda value
    test_results{1, 5} = result;                    % Evaluation result

    %% Convert results to a table and display

    test_results_table = cell2table(test_results, ...
        'VariableNames', {'Method', 'ActivationFunction', 'KValue', 'Lambda', ...
        'TestResult'});  % Convert the cell array to a table with column names

    % Display the test result table
    fprintf('Test result for Neural Network:\n');
    disp(test_results_table);  % Show the table with test results

end
