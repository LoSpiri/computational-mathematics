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

X = cup_x_train(1:120, :);
Y = cup_y_train(1:120, :);

% For testing NN

% X = monks2_x_train;
% Y = monks2_y_train;
% X = cup_x_train;
% Y = cup_y_train;

% Divide X and Y in train and validation sets
[train_X, train_Y, validation_X, validation_Y]=createValidation(X, Y, 0.8);

% test_X = monks2_x_test;
% test_Y = monks2_y_test;
test_X = cup_x_test;
test_Y = cup_y_test;

%% Parameters Initialization

modelParams=modelParameters();
deflectedParams=deflectedParameters();

%% Grid search

% Set the random number generator seed
rng(17);

[results, W1, W2, W1_train, W2_train] = grid_search(train_X, train_Y, validation_X, ...
                                 validation_Y, modelParams, deflectedParams);

%% Method Analysis

% Plot information about time execution
plot_metric=false;
% Plot informations about method
plot_graphs=true;
% Plot how method works in 2D
plot_2D_method=false;

sorted_results = sort_cell_matrix_by_column(results, 11, true);
display_results_method(sorted_results(:, 1:13), X, Y, plot_metric, plot_graphs, plot_2D_method);

%% NN Analysis

% Sort results by Evaluation and display it
sorted_results = sort_cell_matrix_by_column(results, 12, true);

% Plot information about NN
plot_metric=false;
% Plot information about all the results
plot_summary=false;
display_results_NN(sorted_results, plot_metric, plot_summary);

% Show results on test set 
test_results = testDeflected(sorted_results(1, 1:end), test_X, test_Y, W1, W2);
