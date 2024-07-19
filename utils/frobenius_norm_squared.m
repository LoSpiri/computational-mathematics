function frobenius_norm_squared = frobenius_norm_squared(A)
    % Calcola la norma di Frobenius di una matrice A
    % Input: 
    %   A - matrice di dimensione m x n
    % Output:
    %   frobenius_norm - la norma di Frobenius della matrice A
    
    % Calcola la somma dei quadrati di tutti gli elementi della matrice
    frobenius_norm_squared = sum(sum(A.^2));
end

