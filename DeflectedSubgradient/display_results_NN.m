function display_results_NN(results, plot_results, plot_summary)


    % Convert results cell array to table for better visualization
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', ... 
                            'KValue', 'Lambda', 'Rho', 'R', 'Delta', 'MaxIter', ...
                            'MinAlpha', 'Status', 'ElapsedTime', 'Evaluation', ...
                            'ValidationEvaluation', 'ValuesArrays', 'Temp'});

    % Remove duplicate rows
    results_to_plot = removevars(results_table, {'ValuesArrays', 'Temp'});
    [~, unique_idx] = unique(results_to_plot, 'rows', 'stable');
    results_table = results_table(unique_idx, :);
    
    if plot_summary
        % Display the results table
        disp('Results Summary:');
        disp(results_table(:, 1:12));
    end

    % Find the best result based on evaluation metric (lower is better)
    [~, best_idx] = min(results_table.ValidationEvaluation);
    best_result = results_table(best_idx, :);
    fprintf('Best Configuration for Neural Network:\n');
    disp(best_result(:, 1:12));
    
    % Filter the table to keep rows where the hyperparameters match the best result
    if plot_results == true        
        filtered_table = results_table(...
            strcmp(results_table.ActivationFunction, best_result.ActivationFunction) & ...
            results_table.Lambda == best_result.Lambda & ...
            results_table.Temp ==best_result.Temp, :);

        % Sort filtered table by ValidationEvaluation to keep the minimum value for each KValue
        % [~, min_idx] = sort(filtered_table.ValidationEvaluation, 'ascend');
        % filtered_table = filtered_table(min_idx, :);

        plot_metric(filtered_table, 'KValue', {'Evaluation', 'ValidationEvaluation'}, 'K Value', 'Validation and Evaluation', 'Loss by K values during Model execution')
    end
end