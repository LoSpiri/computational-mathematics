function display_results_method(results, plot_results)


    % Convert results cell array to table for better visualization
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', 'KValue', ...
                                    'Lambda', 'Rho', 'R', 'Delta', 'MaxIter', 'Status', ...
                                    'ElapsedTime', 'Evaluation', 'ValidationEvaluation'});

    % Remove duplicate rows
    results_table = unique(results_table, 'rows', 'stable');

    % Find the best result based on evaluation metric (lower is better)
    [~, best_idx] = min(results_table.Evaluation);
    best_result = results_table(best_idx, :);
    fprintf('Best Configuration for Deflected Subgradient:\n');
    disp(best_result);

    if plot_results == true
        
        disp("ciao");
    
    end
end