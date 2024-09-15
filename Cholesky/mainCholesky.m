clear variables;
addpath("activation_functions")
addpath("utils")
addpath("datasets")

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

% Selecting monks1 training set
X = monks2_x_train;
Y = monks2_y_train;
%X = cup_x_train;
%Y = cup_y_train;

% Divide training set in training and validation set
N = size(X, 1);
train_size = floor(0.8 * N); 
train_X = X(1:train_size, :);
validation_X = X(train_size+1:end, :);
train_Y = Y(1:train_size, :);
validation_Y = Y(train_size+1:end, :);

% Selecting monks1 test set
test_X = monks2_x_test;
test_Y = monks2_y_test;
%test_X = cup_x_test;
%test_Y = cup_y_test;

% Store size of the sets
[train_X_r, train_X_c] = size(train_X);
[validation_X_r, ~] = size(validation_X);
[test_X_r, ~] = size(test_X);

% Set the random number generator seed
rng(17);

% Initialize params as a struct
params = struct();

% Assign values to the fields of params
params.activation_functions = {@relu, @tanh, @sigmoid, @identity};
params.activation_functions_names = {'relu', 'tanh', 'sigmoid', 'identity'};
params.k_values = [6, 12, 50, 100, 125, 140];
params.lambda_values = [1e-3, 5e-3, 1e-4, 3e-4];

% Run grid search
[results, W1, W2] = grid_search_Cholesky(train_X, train_Y, train_X_r, train_X_c, ...
                                         validation_X, validation_Y, ...
                                         validation_X_r, params);

% Sort results by Evaluation and display it
sorted_results = sort_cell_matrix_by_column(results, 6, true);
display_results_Cholesky(sorted_results);

% Save hyperparameters of the best configuration
activation_func = sorted_results{1, 1};
layer_dim = sorted_results{1, 2};
lambda = sorted_results{1, 3};

comparation_table = methods_comparation(train_X, train_Y, train_X_r, train_X_c, ...
                                W1, activation_func, layer_dim, lambda);

% Show results on test set and 
test_results = test_Cholesky(test_X, test_Y, test_X_r, train_X_c, ...
                             W1, W2, activation_func, layer_dim, lambda);
