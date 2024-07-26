classdef DeflectedSubgradient
    %DEFLECTEDSUBGRADIENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
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
    end
    
    methods
        function obj = DeflectedSubgradient(W2, delta, rho, R, max_iter, U, D, lambda)
            %DEFLECTEDSUBGRADIENT Construct an instance of this class
            %   Detailed explanation goes here
            obj.W2 = W2;
            obj.delta = delta;
            obj.rho = rho;
            obj.R = R;
            obj.max_iter = max_iter;
            obj.U = U;
            obj.D = D;
            obj.lambda = lambda;
            obj.f_ref = obj.compute_f(obj.W2);
            obj.N=size(U,1);
        end

        function [x_opt, status] = compute_deflected_subgradient(obj)
            f_bar = obj.f_ref;
            f_x = obj.f_ref;
            r = 0;
            x_i = obj.W2;
            for i = 1:obj.max_iter
                g_i = obj.compute_subgradient();
                % -- stopping criteria --
                if sqrt(frobenius_norm_squared(g_i)) < 1e-12
                   status = 'optimal';
                   break;
                end
                % -- --
                if i == 1
                    beta_i = 1;
                    d_i = g_i;
                else
                    gamma_i = obj.update_gamma(g_i, d_i);
                    %
                    disp(gamma_i);
                    % TODO beta_i func
                    beta_i = gamma_i;
                    d_i = gamma_i * g_i + (1 - gamma_i) * d_i;
                end
                alpha_i = obj.update_alpha(beta_i, f_x, d_i);
                %
                disp(alpha_i);
                x_i = x_i - alpha_i * d_i;
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
                status = 'stopped';
                % fprintf('g_i is %g.\n', g_i);
                % fprintf('d_i is %d.\n', d_i);
                % fprintf('x_i is %x.\n', x_i);
            end
            x_opt = x_i;
        end
    end

    methods (Access = private)
        % Metodi privati della classe
        function f_x = compute_f(obj, X)
            L_m = frobenius_norm_squared(obj.U * X - obj.D);
            L_r = norm1(X);
            % Calcola il fattore di normalizzazione
            normalization_factor = 1 / (2 * obj.N);
            % Calcola il valore della funzione di riferimento
            f_x = normalization_factor * L_m + obj.lambda * L_r;
        end

        function g = compute_subgradient(obj)
            % Calcola il numero di campioni N e le dimensioni della matrice W2
            k = size(obj.U, 2);
            m = size(obj.W2, 2);
        
            % Inizializza il gradiente L_m
            grad_Lm = zeros(k, m);
        
            % Calcola il gradiente di L_m
            for t = 1:obj.N
                ht = obj.U(t, :) * obj.W2 - obj.D(t, :); % Calcola ht
                grad_Lm = grad_Lm + obj.U(t, :)' * ht; % Aggiorna il gradiente
            end
            grad_Lm = grad_Lm / obj.N;
        
            % Calcola il subgradiente di L1
            grad_L1 = zeros(k, m);
            for i = 1:k
                for j = 1:m
                    if obj.W2(i, j) > 0
                        grad_L1(i, j) = 1;
                    elseif obj.W2(i, j) < 0
                        grad_L1(i, j) = -1;
                    else
                        grad_L1(i, j) = 2 * rand() - 1;
                    end
                end
            end
        
            % Calcola il subgradiente g
            g = grad_Lm + obj.lambda * grad_L1;
        end

        function alpha_i = update_alpha(obj, beta_i, f_x, d_i)
            norm_d_i = frobenius_norm_squared(d_i);
            num = f_x - obj.f_ref + obj.delta;
            alpha_i = beta_i * (num / norm_d_i);
        end
    end

    methods (Static, Access = private)
        function gamma_i = update_gamma(g_i, d_i)
            % Calcola la norma 2 di g_i e d_i
            norm_g_i = frobenius_norm_squared(g_i);
            norm_d_i = frobenius_norm_squared(d_i);

            % Calcola il prodotto scalare
            dot_product = sum(sum(g_i .* d_i));

            % Calcola gamma_i secondo l'equazione fornita
            gamma_i = (norm_d_i - dot_product) / (norm_g_i + norm_d_i - 2 * dot_product);

            % Assicurati che gamma_i sia nell'intervallo (0, 1)
            gamma_i = max(0, min(1, gamma_i));

            %TODO Put warning with check gamma (0,1)
        end
    end

end

