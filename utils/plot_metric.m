function plot_metric(results_table, x_metric, metrics, x_label, y_label, plot_title, plot_best_only)
    % Set default value for plot_best_only if not provided
    if nargin < 7
        plot_best_only = false;  % Default is to plot all rows
    end

    % Sort the results_table by the x_metric
    results_table = sortrows(results_table, x_metric);  % Sort by x_metric

    % Initialize variables to store best rows if needed
    best_rows = [];

    % Loop through the metrics to find the best results if plot_best_only is true
    if plot_best_only
        for i = 1:numel(metrics)
            best_row = results_table(results_table.(metrics{i}) == min(results_table.(metrics{i})), :);
            best_rows = [best_rows; best_row];
        end
    else
        % If not plotting only the best, use the full table
        best_rows = results_table;
    end

    % Create a figure for plotting the specified metrics
    figure;
    hold on;

    % Colors for the plots
    colors = lines(numel(metrics));  % Generate distinct colors

    % Plot results for each metric
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
