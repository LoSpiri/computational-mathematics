classdef DeflectedSubgradient
    %DEFLECTEDSUBGRADIENT2 Class for performing Deflected Subgradient optimization.
    
    properties
        A
        b
        interval_x1
        interval_x2
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
        Plotf
        elapsed_time
    end
    
    methods
        function obj = DeflectedSubgradient(A, b, interval_x1, interval_x2, W2, delta, rho, R, max_iter, U, D, lambda, Plotf)
            obj.A = A;
            obj.b = b;
            obj.interval_x1 = interval_x1;
            obj.interval_x2 = interval_x2;
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
            obj.Plotf = Plotf;
        end

        function [x_opt, obj, best_result] = compute_deflected_subgradient(obj)
            tic;
            f_bar = obj.f_ref;
            f_x = obj.f_ref;
            r = 0;
            x_i = obj.W2;
            f_values = zeros(1, obj.max_iter);
            norm_g_values = zeros(1, obj.max_iter);
            alpha_values = zeros(1, obj.max_iter);
            gamma_values = zeros(1, obj.max_iter);
            best_result = struct('Iteration', 0, 'FunctionValue', f_x);

            figure;
            obj.plot_surface();
        
            for i = 1:obj.max_iter
                g_i = obj.compute_subgradient(x_i);
                h = norm(obj.U * x_i - obj.D, 2) / (2 * obj.N) + obj.lambda * norm(x_i, 2);
                f_values(i) = h;
                norm_g_values(i) = sqrt(frobenius_norm_squared(g_i));
        
                if sqrt(frobenius_norm_squared(g_i)) < 1e-12
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
                end
                alpha_i = obj.update_alpha(beta_i, f_x, d_i);
                alpha_values(i) = alpha_i;
                
                old_x_i = x_i;
                x_i = x_i - alpha_i * d_i;
                obj.plot_line(old_x_i, x_i);
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
            end
            x_opt = x_i;
            obj.elapsed_time = toc;

            if obj.Plotf == 2
                title('DeflectedSubgradient descent');
                obj.plot_results(f_values, norm_g_values, alpha_values, gamma_values);
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

        function plot_results(obj, f_values, norm_g_values, alpha_values, gamma_values)
            figure;
            subplot(4,1,1);
            plot(f_values, 'LineWidth', 2);
            title('Function Value over Iterations');
            xlabel('Iteration');
            ylabel('f(x)');
        
            subplot(4,1,2);
            plot(norm_g_values, 'LineWidth', 2);
            title('Norm of Subgradient over Iterations');
            xlabel('Iteration');
            ylabel('||g(x)||');
        
            subplot(4,1,3);
            plot(alpha_values, 'LineWidth', 2);
            title('Step Size over Iterations');
            xlabel('Iteration');
            ylabel('alpha');
        
            subplot(4,1,4);
            plot(gamma_values, 'LineWidth', 2);
            title('Gamma over Iterations');
            xlabel('Iteration');
            ylabel('gamma');
        end

        function plot_surface(obj)
            % Define the reduced cost function for plotting
            function f = cost(x)
                A_proj = obj.A(:, 1:2); % Use only the first two columns for projection
                f = 0.5 * x' * (A_proj' * A_proj) * x - obj.b(1:2)' * x; % Adjust to 2D projection
            end
        
            [XX, YY] = meshgrid(obj.interval_x1, obj.interval_x2);
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
        end

        function [] = plot_line(obj, x1, x2)
            PXY = [x1, x2];
            line('XData', PXY(1 , :), 'YData', PXY(2 , :), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'o', 'Color', 'black');
        end

    end

    methods (Static, Access = private)
        % function gamma_i = update_gamma(g_i, d_i)
        %     norm_g_i = frobenius_norm_squared(g_i);
        %     norm_d_i = frobenius_norm_squared(d_i);
        %     dot_product = sum(sum(g_i .* d_i));
        % 
        %     % gamma_i = (norm_d_i - dot_product) / (norm_g_i + norm_d_i - 2 * dot_product);
        % 
        %     % Calculate the denominator
        %     denominator = norm_g_i + norm_d_i - 2 * dot_product;
        % 
        %     eps = 0.00000001;
        %     % Check if the denominator is too close to zero
        %     if abs(denominator) < eps
        %         % Handle near-zero denominator by setting gamma_i to a default value
        %         gamma_i = 0.5;  % You can set a reasonable default value
        %     else
        %         % Compute gamma_i
        %         gamma_i = (norm_d_i - dot_product) / denominator;
        % 
        %         % Ensure gamma_i is within the range [0, 1]
        %         if gamma_i < 0
        %             gamma_i = 0.1;  % Set to 0.1 if it's less than 0
        %         elseif gamma_i > 1
        %             gamma_i = 0.9;  % Set to 0.9 if it's greater than 1
        %         end
        %     end
        % end

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
    end
end

% Helper functions
function norm_sq = frobenius_norm_squared(matrix)
    norm_sq = sum(sum(matrix.^2));
end

function norm1 = norm1(matrix)
    norm1 = sum(sum(abs(matrix)));
end
