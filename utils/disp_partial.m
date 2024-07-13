function disp_partial(var, varName, numResults)
    % var: la variabile da visualizzare
    % varName: il nome della variabile (stringa)
    % numResults: il numero di risultati da visualizzare

    if nargin < 3
        numResults = 5; % valore predefinito
    end

    disp(['Visualizzazione dei primi ', num2str(numResults), ' risultati di ', varName, ':']);
    
    % Verifica se la variabile è una tabella o una matrice
    if istable(var)
        disp(var(1:numResults, :));
    elseif ismatrix(var)
        disp(var(1:min(numResults, size(var, 1)), :));
    else
        disp('Tipo di variabile non supportato.');
    end
end
