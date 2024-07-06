function [cup_x_train, cup_y_train, cup_x_test, cup_y_test] = load_dataset_cup(cup_filename)
    % Caricamento del dataset CUP
    cup_train = readtable(cup_filename, 'FileType', 'text');
    
    % Suddivisione dei dati in training e test
    cup_x_train = table2array(cup_train(1:1300, 1:20));
    cup_y_train = table2array(cup_train(1:1300, 21:22));

    cup_x_test = table2array(cup_train(1301:end, 1:20));
    cup_y_test = table2array(cup_train(1301:end, 21:22));
end
