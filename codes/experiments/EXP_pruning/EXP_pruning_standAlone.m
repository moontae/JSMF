function EXP_pruning_standAlone(data_folder, dataset, p, K, output_base)
    % In command syntax, both n and nprocs are strings.
    if ischar(K) == 1    
        % Convert a string argument to numeric variable.
        K = str2double(K);  
    end
    
    if ischar(p) == 1
        p = str2double(p);
    end

    % Print the information and call the function.
    fprintf('Running Matlab stand alone for EXP_pruning...\n');
    fprintf('- Data folder: %s\n', data_folder);
    fprintf('- Dataset: %s (P = %.2f, K = %d)\n', dataset, p, K);
    fprintf('- Output base: %s\n', output_base);
    EXP_pruning(data_folder, dataset, p, K, output_base);

    % Based on the environment,
    if isdeployed     
        % Standalone mode
        exit;
    else
        % Otherwise,
        close all
    end
end   
