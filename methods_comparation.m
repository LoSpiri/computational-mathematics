function comparation_table=methods_comparation(X, Y, X_r, X_c, W1, activation_func, ...
                                        k, lambda)

    % Convert the activation function string to a function handle
    activation_function = str2func(activation_func);

    comparation_table=cell(3, 6);

    % Initialize the neural network with W1
    comp_nn = NeuralNetwork(X, k, X_r, X_c, W1);
    comp_nn = comp_nn.firstLayer(activation_function);
    comp_nn = comp_nn.secondLayer(size(Y, 2));

    I=eye(size(comp_nn.U,2));
    I=2*X_r*lambda*I;
    A = [comp_nn.U; I];
    B = [Y; zeros(size(comp_nn.U,2), size(Y,2))];
    
    %W2_QR = QR_method(A, B);
    tic;
    [Q, R] = qr(A);
    W2_qr = R \ (Q' * B);
    elapsedTime=toc;

    residual = A * W2_qr - B;
    frob_norm_squared = sum(sum(residual.^2));
    result = (1 / (2 * X_r)) * frob_norm_squared;

    comparation_table{1, 1} = 'QR (Matlab)';
    comparation_table{1, 2} = activation_func;
    comparation_table{1, 3} = k;
    comparation_table{1, 4} = lambda;
    comparation_table{1, 5} = elapsedTime;
    comparation_table{1, 6} = result;

    tic;
    [U, S, V] = svd(A, 'econ'); 
    W2_svd = V * (S \ (U' * B)); 
    elapsedTime = toc;

    residual = A * W2_svd - B;
    frob_norm_squared = sum(sum(residual.^2));
    result = (1 / (2 * X_r)) * frob_norm_squared;

    comparation_table{2, 1} = 'SVD (Matlab)';
    comparation_table{2, 2} = activation_func;
    comparation_table{2, 3} = k;
    comparation_table{2, 4} = lambda;
    comparation_table{2, 5} = elapsedTime;
    comparation_table{2, 6} = result;

    tic; 
    AtA = A' * A;  
    AtB = A' * B;  
    R = chol(AtA); 
    Y = R' \ AtB;  
    W2_chol = R \ Y; 
    elapsedTime = toc; 

    residual = A * W2_chol - B;
    frob_norm_squared = sum(sum(residual.^2));
    result = (1 / (2 * X_r)) * frob_norm_squared;

    comparation_table{3, 1} = 'Cholesky (Matlab)';
    comparation_table{3, 2} = activation_func;
    comparation_table{3, 3} = k;
    comparation_table{3, 4} = lambda;
    comparation_table{3, 5} = elapsedTime;
    comparation_table{3, 6} = result;

    % Convert cell array to table with column names
    comparation_table_view = cell2table(comparation_table, ...
        'VariableNames', {'Method', 'ActivationFunction', 'KValue', 'Lambda', ...
        'ElapsedTime', 'Evaluation'});

    display(comparation_table_view);

end