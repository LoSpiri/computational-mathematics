function display_results(results, plot_results)
    % Convert results cell array to table for better visualization
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', 'KValue', 'Delta', 'Rho', 'R', 'Lambda', 'MaxIter', 'ElapsedTime', 'Evaluation', 'ValidationEvaluation', 'Status'});

    % Display the results table
    disp('Results Summary:');
    disp(results_table);

    % Find the best result based on evaluation metric (lower is better)
    [~, best_idx] = min(results_table.Evaluation);
    best_result = results_table(best_idx, :);
    fprintf('Best Configuration:\n');
    disp(best_result);

    if plot_results == true
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
end