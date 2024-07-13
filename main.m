clear variables;

addpath("activation_functions")
addpath("utils")
addpath("const")

config;

% Definisci i nomi dei file
monks1_train_filename = 'datasets/monks/monks-1.train';
monks1_test_filename = 'datasets/monks/monks-1.test';
monks2_train_filename = 'datasets/monks/monks-2.train';
monks2_test_filename = 'datasets/monks/monks-2.test';
monks3_train_filename = 'datasets/monks/monks-3.train';
monks3_test_filename = 'datasets/monks/monks-3.test';
cup_filename = 'datasets/cup/ml-cup.csv';

% Carica i dataset usando la funzione load_datasets_monks
[monks1_x_train, monks1_y_train, monks1_x_test, monks1_y_test] = load_dataset_monks(monks1_train_filename, monks1_test_filename);
[monks2_x_train, monks2_y_train, monks2_x_test, monks2_y_test] = load_dataset_monks(monks2_train_filename, monks2_test_filename);
[monks3_x_train, monks3_y_train, monks3_x_test, monks3_y_test] = load_dataset_monks(monks3_train_filename, monks3_test_filename);
[cup_x_train, cup_y_train, cup_x_test, cup_y_test] = load_dataset_cup(cup_filename);

% Visualizza i primi risultati di ogni variabile
% disp_partial(monks1_x_train, 'monks1_x_train', 5);
% disp_partial(monks1_y_train, 'monks1_y_train', 5);
% disp_partial(monks2_x_train, 'monks2_x_train', 5);
% disp_partial(monks2_y_train, 'monks2_y_train', 5);
% disp_partial(monks3_x_train, 'monks3_x_train', 5);
% disp_partial(monks3_y_train, 'monks3_y_train', 5);
% disp_partial(cup_x_train, 'cup_x_train', 5);
% disp_partial(cup_y_train, 'cup_y_train', 5);

% Grid search parameters
activation_functions;
activation_functions_names;
W1_c;

% Store grid search results
num_combinations = length(activation_functions) * length(W1_c);
results = cell(num_combinations, 3);

% Index for storing results
index = 1;

% Matrix to test

X = monks1_x_train;

for i = 1:length(activation_functions)
    for w1_c = W1_c
        activation_function = activation_functions{i};
        activation_function_name = activation_functions_names{i};
        [X_r, X_c] = size(X);

        % Start timer
        tic;
        
        % Initialize NN
        nn = NeuralNetwork(X_c, w1_c);
        % Perform the forward pass
        output = nn.activate(X, activation_function);

        elapsed_time = toc;
        % Stopped timer
        
        % Display output
        % disp_partial(output, "output", 5)

        % Store results in cell array
        results{index, 1} = activation_function_name;
        results{index, 2} = w1_c;
        results{index, 3} = elapsed_time;

        index = index + 1;
    end
end

sorted_results = sort_cell_matrix_by_column(results, 3, false);
disp(sorted_results)

% Prendo monks1_x_train, monks1_y_train
% Aggiungerò una colonna di 1 a x_train e creerò la matrice randomica W1
% con bh riga in fondo
% Volendo richiamo tutto z
% Applico funzioni di attivazione a z e lo chiamo u
%
%
% (da mettere dopo) A u aggiungo una riga finale di 1 e moltiplico per W2 + b0 e lo chiamo y
%
%
% Metodi di ottimizzazione: