%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - BATCH_createC_pruning('../../jsmf-dataset/dataset/real_raw', '../../jsmf-dataset/dataset/real_mat_prunning');
%


%%
% Main: BATCH_createC_pruning(input_folder, output_folder)
% 
% Inputs:
%   - input_folder: the folder where docword.txt/vocab.txt/standard.stops files exist.
%   - output_folder: new folder to store the compressed data.
%
% Remark:
%   - This function creates compressed version for each existing training dataset.
%
function BATCH_createC_pruning(input_folder, output_folder, min_objects, min_tokens)
    % Set the default parameters.
    if nargin < 4
        min_tokens = 5;
    end
    if nargin < 3
        min_objects = 3;
    end
    
    % Set the options for the batch work.
    datasets = {'blog', 'yelp', 'nips', 'nytimes', 'movies', 'songs'};
    Ns = [4447, 1606, 5000, 15000, 10000, 10000]; 
    Ps = [0.25, 0.5, 0.75, 1.0]; % Pruning parameters to use
        
    % Create the ouput folder it it does not exist yet.
    if ~isfolder(output_folder)
        mkdir(output_folder);
    end
    
    % Prepare a logger to record the result and performance.
    logger = logging.getLogger('BATCH_createC_pruning_logger', 'path', sprintf('BATCH_createC_pruning.log'));
    logger.info('BATCH_createC_pruning');
    
    for i = 1:length(datasets)        
        % Pick a corresponding option.
        dataset = datasets{i};       
        N = Ns(i);
        logger.info('+ Start creating pruned full co-occurrence dataset for [%s]...', dataset);
        
        bows_filename = sprintf('%s/docword.%s.txt', input_folder, dataset);
        dict_filename = sprintf('%s/vocab.%s.txt', input_folder, dataset);
        stop_filename = sprintf('%s/standard.stops', input_folder);
        
        % For each pruning parameter,
        for p = Ps            
            logger.info('  + Reading and parsing with N=%d, P=%.2f...', N, p);            
            output_filename = sprintf('%s/%s_N-%d_P-%d', output_folder, dataset, N, p*100);            
            [bows, ~] = file.readBows_pruning(bows_filename, dict_filename, stop_filename, N, min_objects, output_filename, p);
            
            % Compress the bag-of-words(objects).
            startTime = tic;
            [C, D1, D2] = file.bows2C(bows, min_tokens);
            elapsedTime = toc(startTime);
            
            % Save the compressed version.
            outputFilename = sprintf('%s/%s_N-%d_P-%d.mat', output_folder, dataset, N, p*100);
            save(outputFilename, 'C', 'D1', 'D2');
            logger.info('    - Saved the original co-occurrence and (co)example frequencies!');
            logger.info('    - Elapsed seconds = %.4f', elapsedTime); 
            
        end
    end
    % Close the logger.
    logging.clearLogger('BATCH_createC_pruning_logger');
    
end
