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

% Grid search parameters
activation_functions;
activation_functions_names;
k;

% Store grid search results
num_combinations = length(activation_functions) * length(W1_c);
results = cell(num_combinations, 3);

% Index for storing results
index = 1;

% Matrix to test
% TODO Testing iteratively
X = monks1_x_train;
Y = monks1_y_train;
[X_r, X_c] = size(X);
[Y_r, Y_c] = size(Y);

W2 = randn(k, Y_c);

DeflectedSubgradient(W2, )

for i = 1:length(activation_functions)
    for w1_c = k
        activation_function = activation_functions{i};
        activation_function_name = activation_functions_names{i};

        % Start timer
        tic;
        
        % Initialize NN
        nn = NeuralNetwork(X_c, w1_c);
        % Perform the forward pass
        U = nn.activate(X, activation_function);

        elapsed_time = toc;
        % Stopped timer
        
        % Display U
        % disp_partial(U, "U", 5)

        % Store results in cell array
        results{index, 1} = activation_function_name;
        results{index, 2} = w1_c;
        results{index, 3} = elapsed_time;

        index = index + 1;
    end
end

sorted_results = sort_cell_matrix_by_column(results, 3, false);
disp(sorted_results)



% Metodi di ottimizzazione:
% 
%
%
%
%
%