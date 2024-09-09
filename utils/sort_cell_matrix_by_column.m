function sorted_matrix = sort_cell_matrix_by_column(cell_matrix, column_index, ascending)
    %Sorts a cell matrix based on a specified column index.
    % 
    % INPUTS:
    %   cell_matrix   - A cell array where each row represents a record and 
    %                   each column represents a field.
    %   column_index  - The index of the column based on which the sorting 
    %                   should be performed.
    %   ascending     - (Optional) A boolean flag indicating whether to sort 
    %                   in ascending order (true) or descending order (false).
    %                   Default is true (ascending order).
    %
    % OUTPUT:
    %   sorted_matrix - The sorted cell matrix based on the specified column.
    
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


