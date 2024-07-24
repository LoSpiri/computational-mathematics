clear variables;

addpath("activation_functions")
addpath("utils")
addpath("const")

% TODO logger

config;
rng(random_state);

% Definisci i nomi dei file
monks1_train_filename = 'datasets/monks/monks-1.train';
monks1_test_filename = 'datasets/monks/monks-1.test';
monks2_train_filename = 'datasets/monks/monks-2.train';
monks2_test_filename = 'datasets/monks/monks-2.test';
monks3_train_filename = 'datasets/monks/monks-3.train';
monks3_test_filename = 'datasets/monks/monks-3.test';
cup_filename = 'datasets/cup/ml-cup.csv';

% Carica i dataset usando la funzione load_datasets_monks
[monks1_x_train, monks1_y_train, monks1_x_test, monks1_y_test] = load_dataset_monks(monks1_train_filename, monks1_test_filename);
[monks2_x_train, monks2_y_train, monks2_x_test, monks2_y_test] = load_dataset_monks(monks2_train_filename, monks2_test_filename);
[monks3_x_train, monks3_y_train, monks3_x_test, monks3_y_test] = load_dataset_monks(monks3_train_filename, monks3_test_filename);
[cup_x_train, cup_y_train, cup_x_test, cup_y_test] = load_dataset_cup(cup_filename);

% Visualizza i primi risultati di ogni variabile
% disp_partial(monks1_x_train, 'monks1_x_train', 5);
% disp_partial(monks1_y_train, 'monks1_y_train', 5);
% disp_partial(monks2_x_train, 'monks2_x_train', 5);
% disp_partial(monks2_y_train, 'monks2_y_train', 5);
% disp_partial(monks3_x_train, 'monks3_x_train', 5);
% disp_partial(monks3_y_train, 'monks3_y_train', 5);
% disp_partial(cup_x_train, 'cup_x_train', 5);
% disp_partial(cup_y_train, 'cup_y_train', 5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Matrix to test
% TODO Testing iteratively
X = monks1_x_train;
Y = monks1_y_train;
[X_r, X_c] = size(X);
[Y_r, Y_c] = size(Y);
disp_partial(X, "X", 5);
disp_partial(Y, "Y", 5);

W2 = randn(k, Y_c);

% Store grid search results
num_combinations = length(activation_functions) * length(k);
results = cell(num_combinations, 3);

% Index for storing results
index = 1;

for i = 1:length(activation_functions)
    % for w1_c = k
    for delta = delta
        for rho = rho
            for R = R
                for lambda = lambda
                    activation_function = activation_functions{i};
                    activation_function_name = activation_functions_names{i};
            
                    % Start timer
                    tic;
                    
                    % Initialize NN
                    nn = NeuralNetwork(X_c, k);
                    % Perform the forward pass
                    U = nn.activate(X, activation_function);
                    % Stopped timer
                    elapsed_time = toc;
                    
                    % Display U
                    % disp_partial(U, "U", 5)
                    ds = DeflectedSubgradient2(W2, delta, rho, R, max_iter, U, Y, lambda, 2);
                    [x_opt, status] = ds.compute_deflected_subgradient();
                    disp(x_opt);
                    disp(status);

                    % Evaluation
                    eval =  norm(U * x_opt - Y, 'fro') / (2 * X_r) + lambda * norm(x_opt, 1);
                    disp_partial(eval, "evaluation", 5);
                    disp_partial(U * x_opt, "U * x_opt", 10);
                    disp_partial(Y, "Y", 10);
            
                    % Store results in cell array
                    results{index, 1} = activation_function_name;
                    results{index, 2} = k;
                    results{index, 3} = elapsed_time;
            
                    index = index + 1;
                end
            end
        end
    end
    % end
end

sorted_results = sort_cell_matrix_by_column(results, 3, false);
disp(sorted_results)