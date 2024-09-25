clear variables;
addpath("../activation_functions")
addpath("../utils")

%% Load Datasets
% Define dataset paths
datasets = struct(...
    'monks1_train', 'datasets/monks/monks-1.train', ...
    'monks1_test', 'datasets/monks/monks-1.test', ...
    'monks2_train', 'datasets/monks/monks-2.train', ...
    'monks2_test', 'datasets/monks/monks-2.test', ...
    'monks3_train', 'datasets/monks/monks-3.train', ...
    'monks3_test', 'datasets/monks/monks-3.test', ...
    'cup', 'datasets/cup/ml-cup.csv');

[monks1_x_train, monks1_y_train, monks1_x_test, monks1_y_test] = load_dataset_monks(datasets.monks1_train, datasets.monks1_test);
[monks2_x_train, monks2_y_train, monks2_x_test, monks2_y_test] = load_dataset_monks(datasets.monks2_train, datasets.monks2_test);
[monks3_x_train, monks3_y_train, monks3_x_test, monks3_y_test] = load_dataset_monks(datasets.monks3_train, datasets.monks3_test);
[cup_x_train, cup_y_train, cup_x_test, cup_y_test] = load_dataset_cup(datasets.cup);

% Testing with monks1 dataset
X = monks1_x_train;
Y = monks1_y_train;
% X = cup_x_train;
% Y = cup_y_train;

%% Training and validation sets
N = size(X, 1);
train_size = floor(0.8 * N); 
train_X = X(1:train_size, :);
validation_X = X(train_size+1:end, :);
train_Y = Y(1:train_size, :);
validation_Y = Y(train_size+1:end, :);

test_X = monks1_x_test;
test_Y = monks1_y_test;
%test_X = cup_x_test;
%test_Y = cup_y_test;

% Store size of the sets
[train_X_r, train_X_c] = size(train_X);
[validation_X_r, ~] = size(validation_X);
[test_X_r, ~] = size(test_X);

%% Set the random number generator seed
rng(17);

%% Parameters Initialization
% Initialize params as a struct
params = struct();

% Assign values to the fields of params
params.activation_functions = {@relu};
params.activation_functions_names = {'relu'};
params.k_values = [2, 4, 8, 16, 32, 64, 128, 256];
params.delta_values = [0.001, 0.01, 0.25, 0.5];
params.rho_values = [0.1, 0.25, 0.5, 0.75, 0.95];
params.R_values = [2, 4, 8, 16, 32, 64, 128, 256];
params.lambda_values = [1e-4, 3e-3, 4e-5];
params.max_iter = [100, 250];

% params.activation_functions = {@relu};
% params.activation_functions_names = {'relu'};
% params.k_values = [2, 4, 8, 16, 32];
% params.delta_values = [0.001, 0.01];
% params.rho_values = [0.1, 0.25, 0.5];
% params.R_values = [2, 8, 32];
% params.lambda_values = [1e-4, 3e-3];
% params.max_iter = [100, 200];

% params.activation_functions = {@relu};
% params.activation_functions_names = {'relu'};
% params.k_values = [4, 16];
% params.delta_values = [0.01];
% params.rho_values = [0.1];
% params.R_values = [8];
% params.lambda_values = [1e-4, 3e-3];
% params.max_iter = [100, 200];

%% Grid search
[results, W1, W2] = grid_search(train_X, train_Y, train_X_r, train_X_c, ...
                      validation_X, validation_Y, ...
                      validation_X_r, params);

% Sort by evaluation and display results
sorted_results = sort_cell_matrix_by_column(results, 10, false);
display_results(sorted_results);

%% Testing
