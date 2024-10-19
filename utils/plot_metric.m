function plot_metric(results_table, x_metric, metrics, x_label, y_label, plot_title)
    % Sort the results_table by KValue
    results_table = sortrows(results_table, x_metric);  % Sort by KValue

    % Initialize variables to store best rows
    best_rows = [];

    % Loop through the metrics to find the best results
    for i = 1:numel(metrics)
        best_row = results_table(results_table.(metrics{i}) == min(results_table.(metrics{i})), :);
        best_rows = [best_rows; best_row];  % Concatenate the best rows
    end

    % Create a figure for plotting the specified metrics
    figure;
    hold on;

    % Colors for the plots
    colors = lines(numel(metrics));  % Generate distinct colors

    % Plot best results for each metric
    for i = 1:numel(metrics)
        plot(best_rows.(x_metric), best_rows.(metrics{i}), 'o-', ...
             'DisplayName', metrics{i}, 'Color', colors(i, :), 'MarkerSize', 10, 'LineWidth', 2);
    end

    % Add labels and legend
    xlabel(x_label);
    ylabel(y_label);
    title(plot_title);
    legend('show');
    grid on;
    hold off;
end
