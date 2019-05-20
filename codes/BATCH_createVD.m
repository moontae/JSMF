%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - BATCH_createVD('../../jsmf-dataset/dataset/real_raw', '../../jsmf-dataset/real_mat_comp');
%


%%
% Main: BATCH_createVD(input_folder, output_folder)
% 
% Inputs:
%   - input_folder: the folder where _train.mat files exist.
%   - output_folder: new folder to store the compressed data.
%
% Remark:
%   - This function creates compressed version for large vocabulary dataset
%     that cannot be processed with the existing algorithms.
%
function BATCH_createVD(input_folder, output_folder, min_objects, min_tokens, T)
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
    
    % Set the options for the batch work. (large vocabs)    
    datasets = {'blog', 'yelp', 'nips', 'nytimes', 'movies', 'songs'};
    Nsets = {[500, 1000, 2000, 4447], ...
             [200, 400, 800, 1606], ...       
             [1250, 2500, 5000, 10000], ...
             [7500, 15000, 30000, 60000], ...
             [1250, 2500, 5000, 10000], ...
             [5000, 10000, 20000, 40000]};               
    Ks = [5, 10, 15, 20, 25, 50, 75, 100, 125, 150];
    Ksets = {Ks(1:8), Ks(1:8), Ks(1:8), Ks(1:end), Ks(1:8), Ks(1:8)};           
    
    datasets = {'blog', 'yelp', 'nips', 'nytimes', 'movies', 'songs'};
    Nsets = {[3000, 4000], ...
             [600, 1000, 1200, 1400, 1600], ...       
             [3750, 6250, 7500, 8750], ...
             [22500, 37500, 45000, 52500], ...
             [3750, 6250, 7500, 8750], ...
             [2500, 7500, 12500, 15000, 17500]};               
    Ks = [5, 10, 15, 20, 25, 50, 75, 100, 125, 150];
    Ksets = {Ks(1:8), Ks(1:8), Ks(1:8), Ks(1:end), Ks(1:8), Ks(1:8)};     
    
%     % Additional settings.
%     datasets = {'nytimes'};
%     Nsets = {[7500, 15000, 30000, 60000]};
%     Ks = [200 250 300];
%     Ksets = {Ks};    

%     % Additional settings.
%     datasets = {'songs'};
%     Nsets = {[5000, 10000, 20000, 40000]};
%     Ks = [200];
%     Ksets = {Ks};   
    
    % Create the ouput folder it it does not exist yet.
    if ~isfolder(output_folder)
        mkdir(output_folder);
    end
    
    % Prepare a logger to record the result and performance.
    logger = logging.getLogger('BATCH_createVD_logger', 'path', sprintf('BATCH_createVD.log'));
    logger.info('BATCH_createVD');
        
    % For each dataset,        
    for i = 1:length(datasets)        
        % Pick a corresponding option.
        dataset = datasets{i};       
        Nset = Nsets{i};
        Kset = Ksets{i};      
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

            % For each number of clusters,
            for K = Kset                       
                logger.info('    + Constructing with K=%d...', K);
                                
                % Compress the bag-of-words(objects) with respect to the current number of clusters.              
                startTime = tic;
                [V, D, H] = file.bows2VD(bows, K, T, min_tokens);             
                elapsedTime = toc(startTime);

                % Save the compressed version.
                outputFilename = sprintf('%s/%s_N-%d_K-%d.mat', output_folder, dataset, N, K);
                save(outputFilename, 'V', 'D', 'H');
                logger.info('      - Saved the rectified and compressd co-occurrence for K=%d!', K);
                logger.info('      - Elapsed seconds = %.4f', elapsedTime);  
            end
        end
    end
    
    % Close the logger.
    logging.clearLogger('BATCH_createVD_logger');
end




%%
% TODO:
%
