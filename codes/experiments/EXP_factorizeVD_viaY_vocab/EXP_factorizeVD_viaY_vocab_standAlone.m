function EXP_factorizeVD_viaY_vocab_standAlone(data_folder, dataset, K, output_base)
    % In command syntax, both n and nprocs are strings.
    if ischar(K) == 1    
        % Convert a string argument to numeric variable.
        K = str2double(K);  
    end

    % Print the information and call the function.
    fprintf('Running Matlab stand alone for EXP_factorizeVD_viaY_vocab...\n');
    fprintf('- Data folder: %s\n', data_folder);
    fprintf('- Dataset: %s (K = %d)\n', dataset, K);
    fprintf('- Output base: %s\n', output_base);
    EXP_factorizeVD_viaY_vocab(data_folder, dataset, K, output_base);

    % Based on the environment,
    if isdeployed     
        % Standalone mode
        exit;
    else
        % Otherwise,
        close all
    end
end   
