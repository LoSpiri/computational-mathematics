clear variables;
addpath("activation_functions")
addpath("utils")

% Define dataset paths
datasets = struct(...
    'monks1_train', 'datasets/monks/monks-1.train', ...
    'monks1_test', 'datasets/monks/monks-1.test', ...
    'monks2_train', 'datasets/monks/monks-2.train', ...
    'monks2_test', 'datasets/monks/monks-2.test', ...
    'monks3_train', 'datasets/monks/monks-3.train', ...
    'monks3_test', 'datasets/monks/monks-3.test', ...
    'cup', 'datasets/cup/ml-cup.csv');

% Load datasets
[monks1_x_train, monks1_y_train, monks1_x_test, monks1_y_test] = load_dataset_monks(datasets.monks1_train, datasets.monks1_test);
[monks2_x_train, monks2_y_train, monks2_x_test, monks2_y_test] = load_dataset_monks(datasets.monks2_train, datasets.monks2_test);
[monks3_x_train, monks3_y_train, monks3_x_test, monks3_y_test] = load_dataset_monks(datasets.monks3_train, datasets.monks3_test);
[cup_x_train, cup_y_train, cup_x_test, cup_y_test] = load_dataset_cup(datasets.cup);

% Selecting monks1 dataset
train_X = monks1_x_train;
train_Y = monks1_y_train;
test_X = monks1_x_test;
test_Y = monks1_y_test;
[train_X_r, train_X_c] = size(train_X);
[test_X_r, test_X_c] = size(test_X);

% Set the random number generator seed
rng(17);

% Initialize params as a struct
params = struct();

% Assign values to the fields of params
params.activation_functions = {@relu, @tanh, @sigmoid, @identity};
params.activation_functions_names = {'relu', 'tanh', 'sigmoid', 'identity'};
params.k_values = [26, 30, 40, 50];
params.lambda_values = [1e-4];

% Run grid search
results = grid_search_Cholesky(train_X, train_Y, train_X_r, train_X_c, params);

% Sort results by Evaluation and display it
sorted_results = sort_cell_matrix_by_column(results, 5, true);
display_results_Cholesky(sorted_results);




