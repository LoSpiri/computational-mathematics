function display_results_NN(results, plot_results)


    % Convert results cell array to table for better visualization
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', 'KValue', ...
                                    'Lambda', 'Rho', 'R', 'Delta', 'MaxIter', 'Status', ...
                                    'ElapsedTime', 'Evaluation', 'ValidationEvaluation'});

    % Remove duplicate rows
    results_table = unique(results_table, 'rows', 'stable');

    % Display the results table
    disp('Results Summary:');
    disp(results_table);

    % Find the best result based on evaluation metric (lower is better)
    [~, best_idx] = min(results_table.ValidationEvaluation);
    best_result = results_table(best_idx, :);
    fprintf('Best Configuration for Neural Network:\n');
    disp(best_result);

    if plot_results == true
        plot_metric(results_table, 'KValue', {'Evaluation', 'ValidationEvaluation'}, 'K Value', 'Validation and Evaluation', 'Loss by K values during Model execution')
    end
end