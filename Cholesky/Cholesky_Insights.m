%% Main Function
function Cholesky_Insights(results, W1, W2, X, Y)
    
    % Print best results
    results_table = cell2table(results, 'VariableNames', {'ActivationFunction', ...
                               'KValue', 'Lambda', 'ElapsedTime', 'Evaluation'});

    fprintf('Best Configuration for Cholesky Method:\n');
    disp(results_table);

end




