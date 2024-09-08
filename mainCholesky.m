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

% Testing with monks1 dataset
train_X = monks1_x_train;
train_Y = monks1_y_train;
test_X = monks1_x_test;
test_Y = monks1_y_test;
[train_X_r, train_X_c] = size(train_X);

% Set the random number generator seed
rng(17);

% Initialize params as a struct
params = struct();

% Assign values to the fields of params
params.activation_functions = {@relu, @tanh};
params.activation_functions_names = {'relu', 'tanh'};
params.k_values = [16];
params.lambda_values = [1e-4];

% Run grid search
results = grid_search_Cholesky(train_X, train_Y, train_X_r, train_X_c, params);

% Sort and display results
sorted_results = sort_cell_matrix_by_column(results, 4, false);  % Sort by Evaluation


%{
display_results(sorted_results);

function display_results(results)
    % Convert results cell array to table for better visualization
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', 'KValue', 'Delta', 'Rho', 'R', 'Lambda', 'MaxIter', 'ElapsedTime', 'Evaluation'});

    % Display the results table
    disp('Results Summary:');
    disp(results_table);

    % Find the best result based on evaluation metric (lower is better)
    [~, best_idx] = min(results_table.Evaluation);
    best_result = results_table(best_idx, :);
    fprintf('Best Configuration:\n');
    disp(best_result);

    % Get unique activation functions
    unique_functions = unique(results_table.ActivationFunction);

    % Plotting results
    % Create a figure for plotting Elapsed Time
    figure;
    hold on;
    colors = lines(numel(unique_functions)); % Distinct colors for each function

    for i = 1:numel(unique_functions)
        % Filter results for the current activation function
        func_results = results_table(strcmp(results_table.ActivationFunction, unique_functions{i}), :);

        % Plot Elapsed Time
        plot(func_results.KValue, func_results.ElapsedTime, 'o-', 'DisplayName', unique_functions{i}, 'Color', colors(i, :));
    end

    % Add labels and legend
    xlabel('K Value');
    ylabel('Elapsed Time (seconds)');
    title('Elapsed Time for Different Activation Functions');
    legend('show');
    grid on;
    hold off;

    % Create a figure for Evaluation
    figure;
    hold on;

    for i = 1:numel(unique_functions)
        % Filter results for the current activation function
        func_results = results_table(strcmp(results_table.ActivationFunction, unique_functions{i}), :);

        % Plot Evaluation
        plot(func_results.KValue, func_results.Evaluation, 'o-', 'DisplayName', unique_functions{i}, 'Color', colors(i, :));
    end

    % Add labels and legend
    xlabel('K Value');
    ylabel('Evaluation');
    title('Evaluation for Different Activation Functions');
    legend('show');
    grid on;
    hold off;
end
%}