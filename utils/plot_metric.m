function plot_metric(results_table, x_metric, metrics, x_label, y_label, plot_title, plot_best_only)
    % PLOT_METRIC Plots specified metrics from a table with options for filtering.
    %
    % This function generates a plot for selected metrics from a given table. The user
    % can specify whether to plot all rows or only the best rows (based on minimum
    % values of each metric). The x-axis is determined by the specified x_metric.
    %
    % INPUT:
    %   - results_table: A table containing the data to be plotted.
    %   - x_metric: The column name (as a string) used for the x-axis.
    %   - metrics: A cell array of strings specifying the column names to be plotted.
    %   - x_label: A string specifying the label for the x-axis.
    %   - y_label: A string specifying the label for the y-axis.
    %   - plot_title: A string specifying the title of the plot.
    %   - plot_best_only (optional): A boolean flag indicating whether to plot all rows
    %     (false, default) or only the rows with the best (minimum) values for each metric (true).
    %
    % OUTPUT:
    %   The function generates a figure and displays the specified plots. It does not
    %   return any values.


    % Set default value for plot_best_only if not provided
    if nargin < 7
        plot_best_only = false;  
    end

    % Sort the results_table by the x_metric
    results_table = sortrows(results_table, x_metric);  % Sort by x_metric

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
    colors = lines(numel(metrics)); 

    for i = 1:numel(metrics)
        plot(best_rows.(x_metric), best_rows.(metrics{i}), 'o-', ...
             'DisplayName', metrics{i}, 'Color', colors(i, :), 'MarkerSize', 10, 'LineWidth', 2);
    end

    xlabel(x_label);         % Label for the x-axis
    ylabel(y_label);         % Label for the y-axis
    title(plot_title);       % Title of the plot
    legend('show');          % Display the legend
    grid on;                 % Enable grid on the plot
    hold off;                % Release the figure for further modifications
end

