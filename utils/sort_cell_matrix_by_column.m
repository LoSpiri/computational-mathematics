function sorted_matrix = sort_cell_matrix_by_column(cell_matrix, column_index, ascending)
    % SORTCELLMATRIXBYCOLUMN Sorts a cell matrix based on a specified column index
    
    % Default ascending order if not specified
    if nargin < 3
        ascending = true;
    end
    
    % Check if column_index is within bounds
    num_columns = size(cell_matrix, 2);
    if column_index < 1 || column_index > num_columns
        error('Column index out of bounds.');
    end
    
    % Convert the column to numeric if possible
    column_data = cell2mat(cell_matrix(:, column_index));
    
    % Determine sorting order
    if ascending
        [~, sorted_indices] = sort(column_data);
    else
        [~, sorted_indices] = sort(column_data, 'descend');
    end
    
    % Sort the cell matrix based on sorted_indices
    sorted_matrix = cell_matrix(sorted_indices, :);
end


