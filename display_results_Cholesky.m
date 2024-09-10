function display_results_Cholesky(results)
    % Display and visualize results from a grid search.
    %
    % INPUT:
    %   results - A cell array containing the results of a grid search with
    %             columns representing Activation Function, K Value, Lambda,
    %             Elapsed Time and Evaluation.
    %
    % This function converts the results to a table, finds the best configuration 
    % based on the evaluation metric, and plots the results for visual analysis.


    % Convert results cell array to table for better visualization
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', 'KValue', ...
        'Lambda', 'ElapsedTime', 'Evaluation', 'Validation_Evaluation' });

    % Display the results table
    disp('Results Summary:');
    disp(results_table);

    % Find the best result based on evaluation metric (lower is better)
    [~, best_idx] = min(results_table.Validation_Evaluation);
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
