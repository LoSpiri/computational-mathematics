function display_results_NN(results, plot_results)


    % Convert results cell array to table for better visualization
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', 'KValue', ...
                                    'Lambda', 'Rho', 'R', 'Delta', 'MaxIter', 'Status', ...
                                    'ElapsedTime', 'Evaluation', 'ValidationEvaluation', 'ValuesArrays', 'Temp'});

    % Remove duplicate rows
    results_no_values_arrays = removevars(results_table, {'ValuesArrays'});
    results_no_values_arrays = removevars(results_no_values_arrays, {'Temp'});
    [~, unique_idx] = unique(results_no_values_arrays, 'rows', 'stable');
    results_table = results_table(unique_idx, :);

    % Display the results table
    disp('Results Summary:');
    disp(results_table(:, 1:11));

    % Find the best result based on evaluation metric (lower is better)
    [~, best_idx] = min(results_table.ValidationEvaluation);
    best_result = results_table(best_idx, :);
    fprintf('Best Configuration for Neural Network:\n');
    disp(best_result(:, 1:11));

    if plot_results == true        % Filter the table to keep rows where the hyperparameters match the best result
        filtered_table = results_table(...
            strcmp(results_table.ActivationFunction, best_result.ActivationFunction) & ...
            results_table.Lambda == best_result.Lambda & ...
            results_table.Temp ==best_result.Temp, :);

        % Sort filtered table by ValidationEvaluation to keep the minimum value for each KValue
        % [~, min_idx] = sort(filtered_table.ValidationEvaluation, 'ascend');
        % filtered_table = filtered_table(min_idx, :);

        plot_loss_by_eval(filtered_table);
    end
end

function plot_loss_by_eval(filtered_table)
    filtered_table = sortrows(filtered_table, 'KValue');
    figure;
    plot(filtered_table.KValue, filtered_table.Evaluation, 'o-', 'LineWidth', 2, 'DisplayName', 'Evaluation');
    hold on;
    plot(filtered_table.KValue, filtered_table.ValidationEvaluation, 'o-', 'LineWidth', 2, 'DisplayName', 'ValidationEvaluation');
    
    xlabel('K Value');
    ylabel('Validation and Evaluation');
    title('Loss by K values during Model execution');
    legend('show');
    grid on;
    hold off;
end