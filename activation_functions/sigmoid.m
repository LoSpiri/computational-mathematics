function output = sigmoid(z)
    % Funzione di attivazione sigmoid
    output = 1 ./ (1 + exp(-z));
end