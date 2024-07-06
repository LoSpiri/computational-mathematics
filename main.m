clear variables;

addpath("activation_functions")
addpath("datasets\*")
addpath("utils")

% Definisci i nomi dei file
monks1_train_filename = 'datasets/monks/monks-1.train';
monks1_test_filename = 'datasets/monks/monks-1.test';
monks2_train_filename = 'datasets/monks/monks-2.train';
monks2_test_filename = 'datasets/monks/monks-2.test';
monks3_train_filename = 'datasets/monks/monks-3.train';
monks3_test_filename = 'datasets/monks/monks-3.test';

% Carica i dataset usando la funzione load_datasets_monks
[monks1_x_train, monks1_y_train, monks1_x_test, monks1_y_test] = load_datasets_monks(monks1_train_filename, monks1_test_filename);
[monks2_x_train, monks2_y_train, monks2_x_test, monks2_y_test] = load_datasets_monks(monks2_train_filename, monks2_test_filename);
[monks3_x_train, monks3_y_train, monks3_x_test, monks3_y_test] = load_datasets_monks(monks3_train_filename, monks3_test_filename);

% Definizione del vettore trasposto x
x = [1; 2; 3];  % Vettore colonna

% Definizione della matrice W
W = [4, 5, 6;
     7, 8, 9;
     10, 11, 12];

% Definizione del vettore b
b = [1; 1; 1];  % Vettore colonna

% Definizione della seconda matrice W2
W2 = [1, 2, 3;
      4, 5, 6;
      7, 8, 9];

% Calcolo del prodotto W * x
Wx = W * x;

% Aggiunta del vettore b
result = Wx + b;

% Elenco delle funzioni di attivazione da testare
activation_functions = {@identity, @sigmoid, @relu, @tanh};
activation_function_names = {'identity', 'sigmoid', 'relu', 'tanh'};

% Iterazione attraverso ciascuna funzione di attivazione
for i = 1:length(activation_functions)
    activation_function = activation_functions{i};
    function_name = activation_function_names{i};
    
    % Misurazione del tempo per l'applicazione della funzione di attivazione
    tic;
    activated_result = activation_function(result);
    elapsed_time = toc;
    
    % Rimoltiplicazione del risultato attivato per la matrice W2
    final_result = W2 * activated_result;
    
    % Visualizzazione del risultato finale e del tempo impiegato
    disp(['Il risultato finale di W2 * (', function_name, '(W * x + b)) è:']);
    disp(final_result);
    disp(['Tempo impiegato per ', function_name, ': ', num2str(elapsed_time), ' secondi']);
end