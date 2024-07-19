function norm1 = norm1(A)
    % Calcola la norma 1 di una matrice A
    % Input:
    %   A - matrice di dimensione m x n
    % Output:
    %   norm1 - la norma 1 della matrice A
    
    % Calcola la somma dei valori assoluti per ogni colonna
    norm1 = sum(abs(A(:)));
end