function display_results_method(results, X, Y, plot_results, plot_graphs)

    % Set default value for plot_graphs if not provided
    if nargin < 5
        plot_graphs = true;
    end

    % Convert results cell array to table for better visualization
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', 'KValue', ...
                                    'Lambda', 'Rho', 'R', 'Delta', 'MaxIter', 'Status', ...
                                    'ElapsedTime', 'Evaluation', 'ValidationEvaluation', 'ValuesArrays'});

    % Remove duplicate rows
    results_no_values_arrays = removevars(results_table, {'ValuesArrays'});
    [~, unique_idx] = unique(results_no_values_arrays, 'rows', 'stable');
    results_table = results_table(unique_idx, :);

    % Find the best result based on evaluation metric (lower is better)
    [~, best_idx] = min(results_table.Evaluation);
    best_result = results_table(best_idx, :);
    fprintf('Best Configuration for Deflected Subgradient:\n');
    disp(best_result);
    best_values_arrays = best_result.ValuesArrays;
    fprintf('Relative Error: %f \n', best_values_arrays.err_values(end));

    % Plot metric if plot_results is true
    if plot_results
        plot_metric(results_table, 'ElapsedTime', {'MaxIter'}, 'Elapsed time', 'Max Iterations', 'Elapsed time during method execution')
    end

    % Plot iteration graphs if plot_graphs is true
    if plot_graphs
        plot_values_by_iteration(best_values_arrays);
        plot_relative_error_by_iteration(best_values_arrays);
        plot_log_relative_error_by_iteration(best_values_arrays);
        plot_descent(best_values_arrays, X, Y);
    end
end


function plot_values_by_iteration(values_arrays)
    figure;
    subplot(4,1,1);
    plot(values_arrays.f_values, 'LineWidth', 2);
    set(gca, 'YScale', 'log');
    title('Function Value over Iterations');
    xlabel('Iteration');
    ylabel('f(x)');

    subplot(4,1,2);
    plot(values_arrays.d_values, 'LineWidth', 2);
    set(gca, 'YScale', 'log');
    title('d over Iterations');
    xlabel('Iteration');
    ylabel('d_i');

    subplot(4,1,3);
    plot(values_arrays.alpha_values, 'LineWidth', 2);
    title('Step Size over Iterations');
    xlabel('Iteration');
    ylabel('alpha');

    subplot(4,1,4);
    plot(values_arrays.gamma_values, 'LineWidth', 2);
    title('Gamma over Iterations');
    xlabel('Iteration');
    ylabel('gamma');
end

function plot_relative_error_by_iteration(values_arrays)
    num_iters = length(values_arrays.err_values);
    iter_numbers = 1:num_iters;
    figure;
    plot(iter_numbers, values_arrays.err_values, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Iteration Number');
    ylabel('Relative Error');
    title('Relative Error vs. Iteration');
    grid on;
end

function plot_log_relative_error_by_iteration(values_arrays)
    num_iters = length(values_arrays.err_values);    
    iter_numbers = 1:num_iters;
    figure;
    plot(iter_numbers, values_arrays.err_values, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    set(gca, 'YScale', 'log');
    xlabel('Iteration Number');
    ylabel('Relative Error');
    title('Relative Error vs. Iteration (Logarithmic Scale)');
    grid on;
end

function plot_descent(values_arrays, X, Y)
    plot_surface(values_arrays.x_values{1}, X, Y);
    
    for i = 1:length(values_arrays.x_values)-1
        plot_line(values_arrays.x_values{i}, values_arrays.x_values{i+1});
    end
end

function plot_surface(x_i, X, Y)
    function f = cost(x, X, Y)
        A_proj = X(:, 1:2);
        b_proj = Y(1:2, :);
        
        if size(b_proj, 2) > 1
            b_proj = sum(b_proj, 2);
        end
        
        f = 0.5 * x' * (A_proj' * A_proj) * x - b_proj' * x;
    end

    % Set a larger initial dynamic range around the initial point x_i
    buffer = 1.0;
    x1_min = x_i(1) - buffer;
    x1_max = x_i(1) + buffer;
    x2_min = x_i(2) - buffer;
    x2_max = x_i(2) + buffer;

    [XX, YY] = meshgrid(linspace(x1_min, x1_max, 50), linspace(x2_min, x2_max, 50));
    X_grid = XX(:); 
    Y_grid = YY(:); 
    Z = zeros(size(X_grid));

    for i = 1:length(X_grid)
        Z(i) = cost([X_grid(i); Y_grid(i)], X, Y);
    end

    ZZ = reshape(Z, size(XX));
    contour(XX, YY, ZZ);
    hold on;

    update_axis_limits([x1_min, x1_max; x2_min, x2_max]);
end


function [] = plot_line(x1, x2)
    PXY = [x1, x2];
    line('XData', PXY(1, :), 'YData', PXY(2, :), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'o', 'Color', 'black');
    
    update_axis_limits([x1, x2]);  
end

function update_axis_limits(points)
    current_x_lim = xlim();
    current_y_lim = ylim();

    x_range = max(points(1, :)) - min(points(1, :));
    y_range = max(points(2, :)) - min(points(2, :));

    margin_factor = 0.05;  % 5% margin
    x_margin = margin_factor * x_range;
    y_margin = margin_factor * y_range;

    min_margin = 0.01;
    if x_margin < min_margin
        x_margin = min_margin;
    end
    if y_margin < min_margin
        y_margin = min_margin;
    end

    new_x_min = min(points(1, :)) - x_margin;
    new_x_max = max(points(1, :)) + x_margin;
    new_y_min = min(points(2, :)) - y_margin;
    new_y_max = max(points(2, :)) + y_margin;

    if new_x_min < current_x_lim(1)
        xlim([new_x_min, current_x_lim(2)]);
    end
    if new_x_max > current_x_lim(2)
        xlim([current_x_lim(1), new_x_max]);
    end
    if new_y_min < current_y_lim(1)
        ylim([new_y_min, current_y_lim(2)]);
    end
    if new_y_max > current_y_lim(2)
        ylim([current_y_lim(1), new_y_max]);
    end
end
