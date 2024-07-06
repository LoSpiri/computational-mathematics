function [monks_x_train, monks_y_train, monks_x_test, monks_y_test] = load_dataset_monks(train_filename, test_filename)
    % Caricamento e conversione dei dati di addestramento
    monks_train = readtable(train_filename, 'FileType', 'text');
    monks_x_train = table2array(monks_train(:, 1:6));
    monks_y_train = table2array(monks_train(:, 7));

    % Caricamento e conversione dei dati di test
    monks_test = readtable(test_filename, 'FileType', 'text');
    monks_x_test = table2array(monks_test(:, 1:6));
    monks_y_test = table2array(monks_test(:, 7));
end
