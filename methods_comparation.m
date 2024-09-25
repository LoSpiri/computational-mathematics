function comparation_table=methods_comparation(best_result, W1, X, Y)


    activation_func=best_result{1,1};
    k = best_result{1,2}; 
    lambda = best_result{1,3};  

    % Convert the activation function string to a function handle
    activation_function = str2func(activation_func);

    comparation_table=cell(3, 6);

    % Initialize the neural network with W1
    comp_nn = NeuralNetwork(X, k, size(X,1), size(X,2), W1);
    comp_nn = comp_nn.firstLayer(activation_function);
    comp_nn = comp_nn.secondLayer(size(Y, 2));

    I=eye(size(comp_nn.U,2));
    I=2*size(X,1)*lambda*I;
    A = [comp_nn.U; I];
    B = [Y; zeros(size(comp_nn.U,2), size(Y,2))];
    
    %% Solve with QR 
    tic;
    [Q, R] = qr(A);
    W2_qr = R \ (Q' * B);
    elapsedTime=toc;

    residual = norm(A * W2_qr - B)/norm(B);

    comparation_table{1, 1} = 'QR (Matlab)';
    comparation_table{1, 2} = activation_func;
    comparation_table{1, 3} = k;
    comparation_table{1, 4} = lambda;
    comparation_table{1, 5} = elapsedTime;
    comparation_table{1, 6} = residual;
    
    %% Solve with SVD 
    tic;
    [U, S, V] = svd(A, 'econ'); 
    W2_svd = V * (S \ (U' * B)); 
    elapsedTime = toc;

    residual = norm(A * W2_svd - B)/norm(B);

    comparation_table{2, 1} = 'SVD (Matlab)';
    comparation_table{2, 2} = activation_func;
    comparation_table{2, 3} = k;
    comparation_table{2, 4} = lambda;
    comparation_table{2, 5} = elapsedTime;
    comparation_table{2, 6} = residual;
    
    %% Solve with Cholesky
    tic; 
    AtA = A' * A;  
    AtB = A' * B;  
    R = chol(AtA); 
    Y = R' \ AtB;  
    W2_chol = R \ Y; 
    elapsedTime = toc; 

    residual = norm(A * W2_chol - B)/norm(B);

    comparation_table{3, 1} = 'Cholesky (Matlab)';
    comparation_table{3, 2} = activation_func;
    comparation_table{3, 3} = k;
    comparation_table{3, 4} = lambda;
    comparation_table{3, 5} = elapsedTime;
    comparation_table{3, 6} = residual;
    
    %% Display results
    % Convert cell array to table with column names
    comparation_table_view = cell2table(comparation_table, ...
        'VariableNames', {'Method', 'ActivationFunction', 'KValue', 'Lambda', ...
        'ElapsedTime', 'Evaluation'});

    display(comparation_table_view);

end