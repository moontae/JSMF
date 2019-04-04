%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - file.BATCH_createVD_large('../../jsmf-raw/dataset/real_raw', '../dataset/real_comp_large');
%


%%
% Main: BATCH_createVD_lage(input_folder, output_folder)
% 
% Inputs:
%   - input_folder: the folder where _train.mat files exist.
%   - output_folder: new folder to store the compressed data.
%
% Remark:
%   - This function creates compressed version for large vocabulary dataset
%     that cannot be processed with the existing algorithms.
%
function BATCH_createVD_lage(input_folder, output_folder, min_objects, min_tokens, T)
    % Set the default parameters.
    if nargin < 5
        T = 50;
    end
    if nargin < 4
        min_tokens = 5;
    end
    if nargin < 3
        min_objects = 3;
    end
    
    
    % Set the options for the batch work.
    datasets = {'nytimes', 'nytimes', 'songs', 'songs'};
    Ns = [30000, 60000, 20000, 40000];
    Ks1 = [5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 250, 300];
    Ks2 = [5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200];
    Ksets = {Ks1, Ks1, Ks2, Ks2};    
        
    % Create the ouput folder it it does not exist yet.
    if ~isfolder(output_folder)
        mkdir(output_folder);
    end
    
    % For each dataset,
    fprintf('[BATCH_createVD_large] Start creating compressed dataset with large vocabulary...\n');
    for i = 1:length(datasets)
        % Pick a corresponding option.
        dataset = datasets{i};       
        N = Ns(i);
        Ks = Ksets{i};
        fprintf('+ Working on [%s] with N=%d...\n', dataset, N);
        
        % Read the input training dataset as it is.
        bowsFilename = sprintf('%s/docword.%s.txt', input_folder, dataset);
        dictFilename = sprintf('%s/vocab.%s.txt', input_folder, dataset);
        stopFilename = sprintf('%s/standard.stops', input_folder, dataset); 
        statFilename = sprintf('%s/%s_N-%d', output_folder, dataset, N);
        [bows, ~] = file.readBows(bowsFilename, dictFilename, stopFilename, N, min_objects, statFilename);                
        
        % For each number of topics,
        for K = Ks                        
            % Compress the bag-of-words(objects).
            [V, D] = file.bows2VD(bows, K, T, min_tokens);             
                    
            % Save the compressed version.
            outputFilename = sprintf('%s/%s_N-%d_K-%d.mat', output_folder, dataset, N, K);
            save(outputFilename, 'V', 'D');
            fprintf('  - Saved the compressd co-occurrence for K=%d!\n', K);
        end
    end
end




%%
% TODO:
%
