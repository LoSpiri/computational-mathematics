classdef DeflectedSubgradient
    
    properties
        A
        b
        W2
        delta
        rho
        R
        max_iter
        f_ref
        U
        D
        lambda
        N
        plot_results
        elapsed_time
    end
    
    methods
        function obj = DeflectedSubgradient(A, b, W2, delta, rho, R, max_iter, U, D, lambda, plot_results)
            obj.A = A;
            obj.b = b;
            obj.W2 = W2;
            obj.delta = delta;
            obj.rho = rho;
            obj.R = R;
            obj.max_iter = max_iter;
            obj.U = U;
            obj.D = D;
            obj.lambda = lambda;
            obj.N = size(obj.U, 1);
            obj.f_ref = obj.compute_f(W2);
            obj.plot_results = plot_results;
        end

        function [x_opt, obj, exit_status] = compute_deflected_subgradient(obj)
            tic;
            f_bar = obj.f_ref;
            f_x = obj.f_ref;
            r = 0;
            x_i = obj.W2;
            f_values = zeros(1, obj.max_iter);
            norm_g_values = zeros(1, obj.max_iter);
            alpha_values = zeros(1, obj.max_iter);
            gamma_values = zeros(1, obj.max_iter);
            d_values = zeros(1, obj.max_iter);
            err_values = zeros(1, obj.max_iter);
            g_bar = 0.048082;
            best_result = struct('Iteration', 0, 'FunctionValue', f_x);
            exit_status = "";

            if obj.plot_results
                figure;
                obj.plot_surface(x_i);
            end
        
            for i = 1:obj.max_iter
                g_i = obj.compute_subgradient(x_i);
                h = norm(obj.U * x_i - obj.D, 2) / (2 * obj.N) + obj.lambda * norm(x_i, 2);
                f_values(i) = h;
                norm_g_values(i) = sqrt(frobenius_norm_squared(g_i));
        
                if sqrt(frobenius_norm_squared(g_i)) < 1e-6
                    exit_status = "STOP CONDITION g_i";
                    break;
                end
        
                if i == 1
                    beta_i = 1;
                    d_i = g_i;
                else
                    gamma_i = obj.update_gamma(g_i, d_i);
                    gamma_values(i) = gamma_i;
                    beta_i = gamma_i;
                    d_i = gamma_i * g_i + (1 - gamma_i) * d_i;
                    d_values(i) = frobenius_norm_squared(d_i);
                end
                if d_i < 1e-6
                    exit_status = "STOP CONDITION d_i";
                    break;
                end
                alpha_i = obj.update_alpha(beta_i, f_x, d_i);
                alpha_values(i) = alpha_i;
                
                old_x_i = x_i;
                x_i = x_i - alpha_i * d_i;

                if obj.plot_results
                    obj.plot_line(old_x_i, x_i);
                end
                
                f_x = obj.compute_f(x_i);
                f_bar = min(f_bar, f_x);
        
                if f_x <= obj.f_ref - obj.delta
                    obj.f_ref = f_bar;
                    r = 0;
                elseif r > obj.R
                    obj.delta = obj.delta * obj.rho;
                    r = 0;
                else
                    r = r + alpha_i * sqrt(frobenius_norm_squared(d_i));
                end
        
                if f_x < best_result.FunctionValue
                    best_result.FunctionValue = f_x;
                    best_result.Iteration = i;
                    best_result.X = x_i;
                end
                
                err_values(i) = abs(sqrt(frobenius_norm_squared(g_i)) - g_bar) / g_bar;
            end
            exit_status = "MAX ITER";
            x_opt = x_i;
            obj.elapsed_time = toc;

            if obj.plot_results
                title('DeflectedSubgradient descent');
                obj.plot(f_values, norm_g_values, alpha_values, gamma_values, d_values, err_values);
            end
        end

        function eval = evaluate_result(obj, x_opt)
            eval = norm(obj.U * x_opt - obj.b, 'fro') / (2 * size(obj.A, 1)) + obj.lambda * norm(x_opt, 1);
        end

    end

    methods (Access = private)
        function f_x = compute_f(obj, X)
            L_m = frobenius_norm_squared(obj.U * X - obj.D);
            L_r = norm1(X);
            normalization_factor = 1 / (2 * obj.N);
            f_x = normalization_factor * L_m + obj.lambda * L_r;
        end

        function g = compute_subgradient(obj, x_i)
            k = size(obj.U, 2);
            m = size(obj.W2, 2);
        
            grad_Lm = zeros(k, m);
        
            for t = 1:obj.N
                ht = obj.U(t, :) * x_i - obj.D(t, :);
                grad_Lm = grad_Lm + obj.U(t, :)' * ht;
            end
            grad_Lm = grad_Lm / obj.N;
        
            grad_L1 = zeros(k, m);
            for i = 1:k
                for j = 1:m
                    if x_i(i, j) > 0
                        grad_L1(i, j) = 1;
                    elseif x_i(i, j) < 0
                        grad_L1(i, j) = -1;
                    else
                        grad_L1(i, j) = 2 * rand() - 1;
                    end
                end
            end
        
            g = grad_Lm + obj.lambda * grad_L1;
        end

        function alpha_i = update_alpha(obj, beta_i, f_x, d_i)
            norm_d_i = frobenius_norm_squared(d_i);
            num = f_x - obj.f_ref + obj.delta;
            alpha_i = beta_i * (num / norm_d_i);

            % Introduce a lower bound for alpha_i
            min_alpha = 1e-2;
            if alpha_i < min_alpha
                alpha_i = min_alpha;
            end
        end

        function plot_surface(obj, x_i)
            % Define the reduced cost function for plotting
            function f = cost(x)
                A_proj = obj.A(:, 1:2); % Use only the first two columns for projection
                f = 0.5 * x' * (A_proj' * A_proj) * x - obj.b(1:2)' * x; % Adjust to 2D projection
            end
        
            % Set a larger initial dynamic range around the initial point x_i
            buffer = 1.0;  % Larger buffer to ensure space for future points
            x1_min = x_i(1) - buffer;
            x1_max = x_i(1) + buffer;
            x2_min = x_i(2) - buffer;
            x2_max = x_i(2) + buffer;
        
            % Create a meshgrid based on the initial point and larger range
            [XX, YY] = meshgrid(linspace(x1_min, x1_max, 50), linspace(x2_min, x2_max, 50));
            X = XX(:); 
            Y = YY(:); 
            Z = zeros(size(X)); % Initialize Z for storing cost values
        
            % Calculate the cost for each (X, Y) pair
            for i = 1:length(X)
                Z(i) = cost([X(i); Y(i)]);
            end
        
            % Reshape Z into the grid for contour plotting
            ZZ = reshape(Z, size(XX));
            contour(XX, YY, ZZ); % Plot the contour
            hold on;
        
            % Set axis limits dynamically based on the initial surface
            obj.update_axis_limits([x1_min, x1_max; x2_min, x2_max]);  % Initial large axis limits
        end


        function [] = plot_line(obj, x1, x2)
            PXY = [x1, x2];
            line('XData', PXY(1, :), 'YData', PXY(2, :), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'o', 'Color', 'black');
            
            % Dynamically update the plot limits to include new points
            obj.update_axis_limits([x1, x2]);  
        end


    end

    methods (Static, Access = private)
        function gamma_i = update_gamma(g_i, d_i)
            v = g_i - d_i;
        
            if all(v == 0)
                gamma_i = 0.5;
            else
                dot_product_v_d = sum(sum(v .* d_i));
                norm_v_squared = frobenius_norm_squared(v);
        
                gamma_i = -dot_product_v_d / norm_v_squared;
        
                if gamma_i < 0 || gamma_i > 1
                    if frobenius_norm_squared(d_i) <= norm_v_squared + 2 * dot_product_v_d + frobenius_norm_squared(d_i)
                        gamma_i = 0.0001;
                    else
                        gamma_i = 0.9999;
                    end
                end
            end
        end

        function plot(f_values, norm_g_values, alpha_values, gamma_values, d_values, err_values)
            figure;
            subplot(5,1,1);
            plot(f_values, 'LineWidth', 2);
            title('Function Value over Iterations');
            xlabel('Iteration');
            ylabel('f(x)');
        
            subplot(5,1,2);
            plot(norm_g_values, 'LineWidth', 2);
            title('Norm of Subgradient over Iterations');
            xlabel('Iteration');
            ylabel('||g(x)||');
        
            subplot(5,1,3);
            plot(alpha_values, 'LineWidth', 2);
            title('Step Size over Iterations');
            xlabel('Iteration');
            ylabel('alpha');
        
            subplot(5,1,4);
            plot(gamma_values, 'LineWidth', 2);
            title('Gamma over Iterations');
            xlabel('Iteration');
            ylabel('gamma');

            subplot(5,1,5);
            plot(d_values, 'LineWidth', 2);
            title('d over Iterations');
            xlabel('Iteration');
            ylabel('d_i');

            % Determine the number of iterations
            num_iters = length(err_values);
            
            % Create an array for iteration numbers (x-axis)
            iter_numbers = 1:num_iters;
        
            % Create a figure for the plot
            figure;
        
            % Plot err_values against iteration numbers
            plot(iter_numbers, err_values, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
        
            % Add labels and title
            xlabel('Iteration Number');
            ylabel('Relative Error');
            title('Relative Error vs. Iteration');
        
            % Add grid for better visibility
            grid on;
        end

        function update_axis_limits(points)
            % Get the current axis limits
            current_x_lim = xlim();
            current_y_lim = ylim();
        
            % Calculate the range of points in x and y directions
            x_range = max(points(1, :)) - min(points(1, :));
            y_range = max(points(2, :)) - min(points(2, :));
        
            % Set a dynamic margin as a percentage of the current range
            margin_factor = 0.05;  % 5% margin
            x_margin = margin_factor * x_range;
            y_margin = margin_factor * y_range;
        
            % Ensure a minimum margin to avoid zero margins
            min_margin = 0.01;
            if x_margin < min_margin
                x_margin = min_margin;
            end
            if y_margin < min_margin
                y_margin = min_margin;
            end
        
            % Calculate new limits with dynamic margins
            new_x_min = min(points(1, :)) - x_margin;
            new_x_max = max(points(1, :)) + x_margin;
            new_y_min = min(points(2, :)) - y_margin;
            new_y_max = max(points(2, :)) + y_margin;
        
            % Update the axis limits only if the new limits expand the current area
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



    end
end

% Helper functions
function norm_sq = frobenius_norm_squared(matrix)
    norm_sq = sum(sum(matrix.^2));
end

function norm1 = norm1(matrix)
    norm1 = sum(sum(abs(matrix)));
end
