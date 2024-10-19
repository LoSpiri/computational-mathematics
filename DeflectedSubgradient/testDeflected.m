function test_results=testDeflected(results, X, Y, W1, W2)

    % Save hyperparameters of the best configuration
    activation_func = results{1, 1};
    k = results{1, 2};
    lambda = results{1, 3};
    
    % Convert the activation function string to a function handle
    activation_function = str2func(activation_func);

    %Create cell
    test_results=cell(1, 5);
                    
    % Initialize the neural network with learned W1 and x_opt
    test_nn = NeuralNetwork(X, k, size(X, 1), size(X,2), W1, W2);
    test_nn = test_nn.firstLayer(activation_function);
    test_nn = test_nn.secondLayer(size(Y, 2));

    %Evaluate test set
    result=test_nn.evaluateModel(Y, test_nn.W2);
    
    test_results{1, 1} = 'DeflectedSubgradient';
    test_results{1, 2} = activation_func;
    test_results{1, 3} = k;
    test_results{1, 4} = lambda;
    test_results{1, 5} = result;

    % Convert cell array to table with column names
    test_results_table = cell2table(test_results, ...
        'VariableNames', {'Method', 'ActivationFunction', 'KValue', 'Lambda', ...
        'TestResult'});
    
    % Display the table
    fprintf('Test result for Neural Network:\n');
    disp(test_results_table);

end