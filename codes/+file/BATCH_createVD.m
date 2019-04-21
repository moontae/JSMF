%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - file.BATCH_createVD('../../jsmf-raw/dataset/real_bows', '../dataset/real_comp');
%


%%
% Main: BATCH_createVD(input_folder, output_folder)
% 
% Inputs:
%   - input_folder: the folder where _train.mat files exist.
%   - output_folder: new folder to store the compressed data.
%
% Remark:
%   - This function creates compressed version for each existing training dataset.
%
function BATCH_createVD(input_folder, output_folder, min_tokens, T)
    % Set the default parameters.
    if nargin < 4
        T = 50;
    end
    if nargin < 3
        min_tokens = 5;
    end
    
    % Set the options for the batch work.
    datasets = {'blog', 'yelp', 'nips', 'nytimes', 'movies', 'songs'};
    Ns = [4447, 1606, 5000, 15000, 10000, 10000];
    Ks1 = [5, 10, 15, 20, 25, 50, 75, 100];
    Ks2 = [5, 10, 15, 20, 25, 50, 75, 100, 125, 150];
    Ksets = {Ks1, Ks1, Ks1, Ks2, Ks1, Ks1};    
    
    % Create the ouput folder it it does not exist yet.
    if ~isfolder(output_folder)
        mkdir(output_folder);
    end
    
    % For each dataset,
    fprintf('[BATCH_createVD] Start creating compressed dataset...\n');
    for i = 1:length(datasets)
        % Pick a corresponding option.
        dataset = datasets{i};       
        N = Ns(i);
        Ks = Ksets{i};
        fprintf('+ Working on [%s] with N=%d...\n', dataset, N);
        
        % Read the input training dataset as it is.
        bowsFilename = sprintf('%s/%s_N-%d_train.bows', input_folder, dataset, N);
        bows = file.readBows(bowsFilename);        
        
        % For each number of topics,
        for K = Ks                        
            % Compress the bag-of-words(objects).
            [V, D] = file.bows2VD(bows, K, T, min_tokens);
                    
            % Save the compressed version.
            outputFilename = sprintf('%s/%s_N-%d_train_K-%d.mat', output_folder, dataset, N, K);
            save(outputFilename, 'V', 'D');
            fprintf('  - Saved the compressd co-occurrence for K=%d!\n', K);
        end
    end
end




%%
% TODO:
%
