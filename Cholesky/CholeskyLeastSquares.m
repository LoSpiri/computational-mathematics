classdef CholeskyLeastSquares
    properties
        N % Number of input rows
        lambda % Regularization Term
        A % Coefficient matrix
        B % Right-hand side matrix
        R % Upper triangular matrix from Cholesky decomposition
        AtA % A transpose times A
        AtB % A transpose times B
        ComputeCholeskyTime % Time taken to compute Cholesky decomposition
    end
    
    methods
        function obj = CholeskyLeastSquares(U, D, lambda)
            % Constructor for CholeskyLeastSquares class
            % Initializes matrices A, B, AtA, and AtB with regularization term
            tic;
            obj.N = size(U, 1);
            obj.lambda=lambda;
            I=eye(size(U,2));
            I=2*obj.N*obj.lambda*I;
            obj.A = [U; I];
            obj.B = [D; zeros(size(U,2), size(D,2))];
            obj.AtA = (obj.A)' * obj.A;
            obj.AtB = (obj.A)' * obj.B;
            obj.ComputeCholeskyTime = toc;
        end
        
        function obj = computeCholesky(obj)
            % Perform manual Cholesky decomposition of AtA
            % Decomposes AtA into an upper triangular matrix R such that AtA = R'R
            tic; % Start timing
            n = size(obj.AtA, 1);
            obj.R = zeros(n);
            for i = 1:n
                for j = i:n
                    if i == j
                        sum = 0;
                        for k = 1:i-1
                            sum = sum + obj.R(k, i)^2;
                        end
                        obj.R(i, i) = sqrt(obj.AtA(i, i) - sum);
                    else
                        sum = 0;
                        for k = 1:i-1
                            sum = sum + obj.R(k, i) * obj.R(k, j);
                        end
                        obj.R(i, j) = (obj.AtA(i, j) - sum) / obj.R(i, i);
                    end
                end
            end
            obj.ComputeCholeskyTime = obj.ComputeCholeskyTime+toc; % End timing
        end
        
        function [x, obj] = solve(obj)
            % Solve the system using the Cholesky factorization
            % R'R = AtA
            % Solve R'y = AtB using forward substitution
            tic; % Start timing
            y = obj.forwardSubstitution(obj.R', obj.AtB);
            % Solve Rx = y using backward substitution
            x = obj.backwardSubstitution(obj.R, y);
            obj.ComputeCholeskyTime =obj.ComputeCholeskyTime +toc; % End timing
        end
        
        function y = forwardSubstitution(~, L, b)
            % Forward substitution to solve Ly = b
            % L is a lower triangular matrix
            n = size(L, 1);
            y = zeros(n, size(b, 2));
            for i = 1:n
                y(i, :) = (b(i, :) - L(i, 1:i-1) * y(1:i-1, :)) / L(i, i);
            end
        end
        
        function x = backwardSubstitution(~, Q, y)
            % Backward substitution to solve Qx = y
            % Q is an upper triangular matrix
            n = size(Q, 1);
            x = zeros(n, size(y, 2));
            for i = n:-1:1
                x(i, :) = (y(i, :) - Q(i, i+1:n) * x(i+1:n, :)) / Q(i, i);
            end
        end

        function objective_value = evaluateResult(obj, x_opt)
            % Evaluate the result and print the objective function value
            residual = obj.A * x_opt - obj.B;
            frob_norm_squared = sum(sum(residual.^2));
            objective_value = (1 / (2 * obj.N)) * frob_norm_squared;
            % fprintf('Objective function value: %f\n', objective_value);
        end

    end
end
