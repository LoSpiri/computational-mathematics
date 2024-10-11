function [train_X, train_Y, val_X, val_Y] = createValidation(X, Y, perc)
    % This function splits the input data (X, Y) into training and validation sets 
    % based on the given percentage (perc).
    
    % INPUT:
    %   X    - Input data matrix.
    %   Y    - Output data matrix (corresponding to X).
    %   perc - Percentage of data to allocate to the training set (as a value between 0 and 1).
    
    % OUTPUT:
    %   train_X - Training set inputs (first perc% of X).
    %   train_Y - Training set outputs (first perc% of Y).
    %   val_X   - Validation set inputs (remaining (1-perc)% of X).
    %   val_Y   - Validation set outputs (remaining (1-perc)% of Y).


    train_size = floor(perc * size(X, 1)); 
    train_X = X(1:train_size, :);
    val_X = X(train_size+1:end, :);
    train_Y = Y(1:train_size, :);
    val_Y = Y(train_size+1:end, :);