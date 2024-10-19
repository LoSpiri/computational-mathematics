clear variables;
addpath("Model/")
addpath("utils/")
addpath("datasets/")

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


%% Training and validation sets

% For testing method

% X = cup_x_train(1:250, :);
% Y = cup_y_train(1:250, :);

% For testing NN

% X = monks1_x_train;
% Y = monks1_y_train;
X = cup_x_train;
Y = cup_y_train;

% Divide X and Y in train and validation sets
[train_X, train_Y, validation_X, validation_Y]=createValidation(X, Y, 0.8);

% test_X = monks1_x_test;
% test_Y = monks1_y_test;
test_X = cup_x_test;
test_Y = cup_y_test;

%% Set the random number generator seed

rng(17);

%% Parameters Initialization

modelParams=modelParameters();
deflectedParams=deflectedParameters();

%% Grid search

plot_results = true;
[results, W1, W2, W1_train, W2_train] = grid_search(train_X, train_Y, validation_X, ...
                                 validation_Y, modelParams, deflectedParams, false);

%% Method Analysis

sorted_results = sort_cell_matrix_by_column(results, 10, true);
display_results_method(sorted_results, plot_results);

%% NN Analysis

% Sort results by Evaluation and display it
sorted_results = sort_cell_matrix_by_column(results, 11, true);
display_results_NN(sorted_results, plot_results);
%display(sorted_results(1, 1:end))

% Show results on test set and 
test_results = testDeflected(sorted_results(1, 1:end), test_X, test_Y, W1, W2);
