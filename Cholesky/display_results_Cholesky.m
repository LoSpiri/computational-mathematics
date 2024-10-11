function display_results_Cholesky(results, plots)
    % Display the grid search results for the Cholesky method and visualize them
    %
    % INPUT:
    %   results - A cell array containing the grid search results with columns
    %             for activation function, K value, lambda, elapsed time, evaluation, 
    %             and validation evaluation.
    %   plots   - A boolean flag that triggers the plotting of results if true.
    %
    % OUTPUT:
    %   None. Displays results and optionally plots them.

    %% Convert results to table and display

    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', 'KValue', ...
        'Lambda', 'ElapsedTime', 'Evaluation', 'Validation_Evaluation' });
    
    disp('Results Summary:');
    disp(results_table);
    
    %Find best configuration
    [~, best_idx] = min(results_table.Validation_Evaluation);
    best_result = results_table(best_idx, :);
    fprintf('Best Configuration for Neural Network:\n');
    disp(best_result);

    %% Plot results if requested

    if plots
        display_Plot_Cholesky(results_table, best_result);
    end
end

function display_Plot_Cholesky(results_table, best_result)
    % Plot the grid search results for elapsed time, evaluation, and validation evaluation
    %
    % INPUT:
    %   results_table - Table containing the grid search results.
    %   best_result   - Table row with the best configuration.
    %
    % OUTPUT:
    %   None. Displays the plots.

    unique_functions = unique(results_table.ActivationFunction);
    % Distinct colors for each function
    colors = lines(numel(unique_functions)); 

    %% Plot Elapsed Time for different activation functions

    figure;
    hold on;
    for i = 1:numel(unique_functions)
        % Filter and sort the results for the current activation function
        func_results = results_table(strcmp(results_table.ActivationFunction, unique_functions{i}) & ...
                                     results_table.Lambda == best_result.Lambda, :);
        func_results = sortrows(func_results, 'KValue');

        % Plot elapsed time
        plot(func_results.KValue, func_results.ElapsedTime, 'o-', 'DisplayName', unique_functions{i}, 'Color', colors(i, :));
    end
    xlabel('K Value');
    ylabel('Elapsed Time (seconds)');
    title('Elapsed Time for Different Activation Functions');
    legend('show');
    grid on;
    hold off;

    %% Plot Evaluation for different activation functions

    figure;
    hold on;
    for i = 1:numel(unique_functions)
        % Filter and sort the results for the current activation function
        func_results = results_table(strcmp(results_table.ActivationFunction, unique_functions{i}) & ...
                                     results_table.Lambda == best_result.Lambda, :);
        func_results = sortrows(func_results, 'KValue');

        % Plot evaluation
        plot(func_results.KValue, func_results.Evaluation, 'o-', 'DisplayName', unique_functions{i}, 'Color', colors(i, :));
    end
    xlabel('K Value');
    ylabel('Evaluation');
    title('Evaluation for Different Activation Functions');
    legend('show');
    grid on;
    hold off;

    %% Plot Validation Evaluation for different activation functions

    figure;
    hold on;
    for i = 1:numel(unique_functions)
        % Filter and sort the results for the current activation function
        func_results = results_table(strcmp(results_table.ActivationFunction, unique_functions{i}) & ...
                                     results_table.Lambda == best_result.Lambda, :);
        func_results = sortrows(func_results, 'KValue');

        % Plot validation evaluation
        plot(func_results.KValue, func_results.Validation_Evaluation, 'o-', 'DisplayName', unique_functions{i}, 'Color', colors(i, :));
    end
    xlabel('K Value');
    ylabel('Validation Evaluation');
    title('Validation Evaluation for Different Activation Functions');
    legend('show');
    grid on;
    hold off;

    %% Compare training and validation losses for the best activation function
    
    figure;
    hold on;
    func_results = results_table(strcmp(results_table.ActivationFunction, best_result.ActivationFunction) & ...
                                 results_table.Lambda == best_result.Lambda, :);
    func_results = sortrows(func_results, 'KValue');

    % Plot training and validation evaluations
    plot(func_results.KValue, func_results.Evaluation, 'o-', 'DisplayName', 'Evaluation', 'Color', 'b');
    plot(func_results.KValue, func_results.Validation_Evaluation, 'o-', 'DisplayName', 'Validation', 'Color', 'r');

    xlabel('K Value');
    ylabel('Loss');
    title('Comparison of Training and Validation Loss');
    legend('show');
    grid on;
    hold off;
end
