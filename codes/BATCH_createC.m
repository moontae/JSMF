%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - BATCH_createC('../../jsmf-dataset/dataset/real_raw', '../../jsmf-dataset/real_mat_full');
%


%%
% Main: BATCH_createC(input_folder, output_folder)
% 
% Inputs:
%   - input_folder: the folder where _train.mat files exist.
%   - output_folder: new folder to store the compressed data.
%
% Remark:
%   - This function creates compressed version for large vocabulary dataset
%     that cannot be processed with the existing algorithms.
%
function BATCH_createC(input_folder, output_folder, min_objects, min_tokens)
    % Set the default parameters.
    if nargin < 4
        min_tokens = 5;
    end
    if nargin < 3
        min_objects = 3;
    end    
    
    % Set the options for the batch work. (large vocabs)    
    datasets = {'blog', 'yelp', 'nips', 'nytimes', 'movies', 'songs'};
    Nsets = {[500, 1000, 2000, 4447], ...
             [200, 400, 800, 1606], ...       
             [1250, 2500, 5000, 10000], ...
             [7500, 15000], ...
             [1250, 2500, 5000, 10000], ...
             [5000, 10000]};               
        
    % Create the ouput folder it it does not exist yet.
    if ~isfolder(output_folder)
        mkdir(output_folder);
    end
    
    % Prepare a logger to record the result and performance.
    logger = logging.getLogger('BATCH_createC_logger', 'path', sprintf('BATCH_createC.log'));
    logger.info('BATCH_createC');
        
    % For each dataset,        
    for i = 1:length(datasets)        
        % Pick a corresponding option.
        dataset = datasets{i};       
        Nset = Nsets{i};         
        logger.info('+ Start creating rectified and compressed datasets for [%s]...', dataset);
        
        % Prepare the data filenames with respect to the current dataset.
        bowsFilename = sprintf('%s/docword.%s.txt', input_folder, dataset);
        dictFilename = sprintf('%s/vocab.%s.txt', input_folder, dataset);
        stopFilename = sprintf('%s/standard.stops', input_folder); 
        
        % For each size of vocabulary,
        for N = Nset
            logger.info('  + Reading and parsing with N=%d...', N);
            
            % Prepare the statistic filename with respect to the current size of vocabulary.            
            statFilename = sprintf('%s/%s_N-%d', output_folder, dataset, N);            
            
            % Read the BOW data with pruning the vocabulary.
            [bows, ~] = file.readBows(bowsFilename, dictFilename, stopFilename, N, min_objects, statFilename);                
                               
            % Compress the bag-of-words(objects) with respect to the current number of clusters.              
            startTime = tic;
            [C, D1, D2] = file.bows2C(bows, min_tokens);             
            elapsedTime = toc(startTime);

            % Save the compressed version.
            outputFilename = sprintf('%s/%s_N-%d.mat', output_folder, dataset, N);
            save(outputFilename, 'C', 'D1', 'D2');
            logger.info('    - Saved the original co-occurrence and (co)example frequencies!');
            logger.info('    - Elapsed seconds = %.4f', elapsedTime);              
        end
    end
    
    % Close the logger.
    logging.clearLogger('BATCH_createC_logger');
end




%%
% TODO:
%
