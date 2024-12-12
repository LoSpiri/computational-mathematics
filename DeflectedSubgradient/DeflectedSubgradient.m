classdef DeflectedSubgradient
    
    properties
        U
        b
        W2
        delta
        rho
        R
        max_iter
        min_alpha
        lambda
        N
        f_ref
        elapsed_time
    end
    
    methods
        %% Initialize properties
        function obj = DeflectedSubgradient(U, b, W2, delta, rho, R, max_iter, min_alpha, lambda)
            obj.U = U;
            obj.b = b;
            obj.W2 = W2;
            obj.delta = delta;
            obj.rho = rho;
            obj.R = R;
            obj.max_iter = max_iter;
            obj.min_alpha=min_alpha;
            obj.lambda = lambda;
            obj.N = size(obj.U, 1);
            obj.f_ref = obj.evaluate_f(W2);
        end
        
        %% Deflected Subgradient Method
        function [x_opt, values_arrays, obj, exit_status] = compute_deflected_subgradient(obj)
           
            % Initialize variables
            tic;
            f_bar = obj.f_ref;
            f_x = obj.f_ref;
            r = 0;
            x_i = obj.W2;
            exit_status = "";

            %Variables for plots
            f_values = zeros(1, obj.max_iter);
            alpha_values = zeros(1, obj.max_iter);
            gamma_values = zeros(1, obj.max_iter);
            d_values = zeros(1, obj.max_iter);
            err_values = zeros(1, obj.max_iter);
            delta_values = zeros(1, obj.max_iter);
            r_values = zeros(1, obj.max_iter);
            x_values = cell(1, obj.max_iter);

            %Optimal result for relative error
            y_bar =  0.017573;

            
            % Start iterating
            for i = 1:obj.max_iter
                
                % Compute subgradient
                g_i = obj.compute_subgradient(x_i);
                if sqrt(frobenius_norm_squared(g_i)) < 1e-6
                    exit_status = "STOP CONDITION g_i";
                    break;
                end
        
                % Compute gamma, beta and d
                if i == 1
                    beta_i = 1;
                    d_i = g_i;
                    gamma_i = 1;
                else
                    gamma_i = obj.update_gamma(g_i, d_i);
                    beta_i = gamma_i;
                    d_i = gamma_i * g_i + (1 - gamma_i) * d_i;
                end

                if d_i < 1e-6
                    exit_status = "STOP CONDITION d_i";
                    break;
                end

                % Compute alpha, x and f(x)
                alpha_i = obj.update_alpha(beta_i, f_x, d_i);
                x_i = x_i - alpha_i * d_i;
                f_x = obj.evaluate_f(x_i);

                % Checks for delta
                f_bar = min(f_bar, f_x);
                if f_x <= obj.f_ref - (obj.delta/2)
                    obj.f_ref = f_bar;
                    r = 0;
                elseif r > obj.R
                    obj.delta = obj.delta * obj.rho;
                    r = 0;
                else
                    r = r + alpha_i * sqrt(frobenius_norm_squared(d_i));
                end
                
                % Save variables for plots
                f_values(i) = f_x;
                alpha_values(i) = alpha_i;
                gamma_values(i) = gamma_i;
                d_values(i) = frobenius_norm_squared(d_i);
                err_values(i) = abs(f_x - y_bar) / y_bar;
                delta_values(i) = obj.delta;
                r_values(i) = r;
                x_values{i} = x_i;
            end

            % Export variables
            exit_status = "MAX ITER";
            values_arrays = struct();
            values_arrays.f_values = f_values;
            values_arrays.alpha_values = alpha_values;
            values_arrays.gamma_values = gamma_values;
            values_arrays.d_values = d_values;
            values_arrays.err_values = err_values;
            values_arrays.delta_values = delta_values;
            values_arrays.r_values = r_values;
            values_arrays.x_values = x_values;
            x_opt = x_i;
            obj.elapsed_time = toc;
        end
        
        %% Computing objective function
        function f_x = evaluate_f(obj, x)
            L_m = frobenius_norm_squared(obj.U * x - obj.b);
            L_r = norm1(x);
            normalization_factor = 1 / (2 * obj.N);
            f_x = normalization_factor * L_m + obj.lambda * L_r;
        end

    end

    methods (Access = private)
        %% Compute subgradient
        function g = compute_subgradient(obj, x_i)
            k = size(obj.U, 2);
            m = size(obj.W2, 2);
            grad_Lm = zeros(k, m);
        
            for t = 1:obj.N
                ht = obj.U(t, :) * x_i - obj.b(t, :);
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
        
        %% Update alpha
        function alpha_i = update_alpha(obj, beta_i, f_x, d_i)
            norm_d_i = frobenius_norm_squared(d_i);
            num = f_x - obj.f_ref + obj.delta;
            alpha_i = beta_i * (num / norm_d_i);

            % Introduce a lower bound for alpha_i
            if alpha_i < obj.min_alpha
                alpha_i = obj.min_alpha;
            elseif alpha_i > 1
                alpha_i=1;
            end
        end
    end

    methods (Static, Access = private)
        %% Update gamma
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
                        gamma_i = 0;
                    else
                        gamma_i = 1;
                    end
                end
            end
        end
    end
end

%% Compute frobenius norm squared
function norm_sq = frobenius_norm_squared(matrix)
    norm_sq = sum(sum(matrix.^2));
end

%% Compute norm1
function norm1 = norm1(matrix)
    norm1 = sum(sum(abs(matrix)));
end
