%% Main Function
function Cholesky_Insights(results, W1, W2, X, Y)
    
    % Print best results
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', ...
                               'KValue', 'Lambda', 'ElapsedTime', 'Evaluation'});

    fprintf('Best Configuration for Cholesky Method:\n');
    disp(results_table);

    % Elapsed time for differente dimension of X
    DifferentRows(results, W1, X, Y);

end


%% Elapsed time for differente dimension of X
function DifferentRows(results, W1, X, Y)

    activation_func = results{1, 1};
    k = results{1, 2};
    
    max_rows = 1036; 
    min_rows = 500; 
    
    num_configurations = 100;
    row_sizes = round(linspace(min_rows, max_rows, num_configurations));
    
    execution_times = zeros(1, num_configurations);
    
    % Esegui il metodo su ogni sottomatrice di X e misura il tempo di esecuzione
    for i = 1:num_configurations
        % tic;
        % Prendi le prime 'row_sizes(i)' righe di X e Y
        X_sub = X(1:row_sizes(i), :);
        Y_sub = Y(1:row_sizes(i), :);

        % execution_times(i)=SolveCholesky(X_sub, Y_sub, W1, activation_func, k);
        execution_times(i)=SolveCholesky(X_sub, Y_sub, W1, activation_func, k);
        %execution_times(i)=toc;
    end

    % Traccia il grafico del tempo di esecuzione
    figure;
    plot(row_sizes, execution_times, '-o');
    xlabel('Numbers of rows of X');
    ylabel('Elapsed Time (s)');
    title('Method execution time as a function of the number of rows of X');
    grid on;
end

%% Function to Solve Cholesky
function elapsedTime=SolveCholesky(X, Y, W1, activation_func_str, k)

    % Convert the activation function string to a function handle
    activation_func = str2func(activation_func_str);

    % Initialize the neural network with learned W1 and x_opt
    nn = NeuralNetwork(X, k, size(X,1), size(X,2), W1);
    nn = nn.firstLayer(activation_func);
    nn = nn.secondLayer(size(Y, 2));
    
    % Initialize and solve the Cholesky least squares problem
    chol = CholeskyLeastSquares(nn.U, Y, 0.0001);
    % Compute the Cholesky decomposition
    chol = chol.computeCholesky();
    % Solve the least squares problem
    [~, chol] = chol.solve();
    elapsedTime = chol.ComputeCholeskyTime;

end