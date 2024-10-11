clear variables;
addpath("activation_functions")
addpath("utils")
addpath("datasets")

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


%% Training , Validation and Test sets

% For testing method

% X = cup_x_train(1:200, :);
% Y = cup_y_train(1:200, :);

% For testing NN

% X = monks3_x_train;
% Y = monks3_y_train;
X = cup_x_train;
Y = cup_y_train;

train_size = floor(0.8 * size(X, 1)); 
train_X = X(1:train_size, :);
validation_X = X(train_size+1:end, :);
train_Y = Y(1:train_size, :);
validation_Y = Y(train_size+1:end, :);

% test_X = monks3_x_test;
% test_Y = monks3_y_test;
test_X = cup_x_test;
test_Y = cup_y_test;

%% Set the random number generator seed
rng(17);

%% Model Parameters Initialization
% Initialize params as a struct
params = struct();

% Assign values to the fields of params for neural network
params.activation_functions = {@relu, @tanh, @sigmoid};
params.activation_functions_names = {'relu', 'tanh', 'sigmoid'};
params.k_values = [50];
params.lambda_values = [5e-4];

%% Grid search

% Find best configuration for NN and for Training Method
[results, W1, W2, W1_train, W2_train] = grid_search_Cholesky(train_X, train_Y, ...
                                                             validation_X, validation_Y, ...
                                                             params);

%% Method analysis

% Sort results by Training Evaluation to take the best configuration
sorted_results_train = sort_cell_matrix_by_column(results, 5, true);

Cholesky_Insights(sorted_results_train(1, 1:end-1), W1_train, train_X, train_Y);

comparation_table = methods_comparation(sorted_results_train(1, 1:end-1), ...
                                                    W1_train, train_X, train_Y);

%% NN analysis
% Sort results by Evaluation and display it
sorted_results = sort_cell_matrix_by_column(results, 6, true);
display_results_Cholesky(sorted_results);

% Show results on test set and 
test_results = test_Cholesky(sorted_results(1, 1:end-1), test_X, test_Y, W1, W2);
