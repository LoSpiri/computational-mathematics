classdef CholeskyLeastSquares
    properties
        A % Coefficient matrix
        B % Right-hand side matrix
        R % Upper triangular matrix from Cholesky decomposition
        AtA % A transpose times A
        AtB % A transpose times B
    end
    
    methods
        function obj = CholeskyLeastSquares(A, B)
            obj.A = A;
            obj.B = B;
            obj.AtA = A' * A;
            obj.AtB = A' * B;
        end
        
        function obj = computeCholesky(obj)
            % Perform manual Cholesky decomposition of AtA
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
        end
        
        function x = solve(obj)
            % Solve the system using the Cholesky factorization
            % R'R = AtA
            % Solve R'y = AtB using forward substitution
            y = obj.forwardSubstitution(obj.R', obj.AtB);
            % Solve Rx = y using backward substitution
            x = obj.backwardSubstitution(obj.R, y);
        end
        
        function y = forwardSubstitution(~, L, b)
            % Forward substitution to solve Ly = b
            n = size(L, 1);
            y = zeros(n, size(b, 2));
            for i = 1:n
                y(i, :) = (b(i, :) - L(i, 1:i-1) * y(1:i-1, :)) / L(i, i);
            end
        end
        
        function x = backwardSubstitution(~, U, y)
            % Backward substitution to solve Ux = y
            n = size(U, 1);
            x = zeros(n, size(y, 2));
            for i = n:-1:1
                x(i, :) = (y(i, :) - U(i, i+1:n) * x(i+1:n, :)) / U(i, i);
            end
        end
    end
end