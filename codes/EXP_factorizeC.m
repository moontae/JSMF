%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Remark:
%   - This is a function that learns given number of topics and their 
%     correlations from a training dataset ending with '_train.mat'.
%   - You can generate a Matlab standalone runtime executable for this
%     function by running experiments/EXP_factorizeC/EXP_factorizeC_make.m.
%   - Then you can submit a series of parallel jobs across differnet
%     datasets and topics by using a job queue in your cluster.
%   - Individual results must be later merged through the python script.
%
% Example: 
%   - EXP_factorizeC('../dataset/real_mat', 'nips_N-5000', 5, '../models/real');
%   
function EXP_factorizeC(data_folder, dataset, K, output_base)        
    % Setup the types of rectifications and optimizations.
    rectifiers = {'Baseline', 'AP', 'DC', 'DP'};    
    %rectifiers = {'AP'};
    optimizers = {'activeSet', 'admmDR', 'expGrad'};
    %optimizers = {'activeSet'};
    
    % Setup the folder of dataset and the base folder to store the outputs.
    % (Note that the runtime executable will be stored on experiments folder
    % from which the relative path must be specified)
    dataFolder = data_folder;
    outputBase = output_base;
    %dataFolder = sprintf('../../dataset/real_mat');
    %outputBase = sprintf('../../models/real');    
        
    % Prepare a logger to record the result and performance.
    logger = logging.getLogger('EXP_factorizeC_logger', 'path', sprintf('EXP_factorizeC_%s_K-%d.log', dataset, K));
    logger.info('EXP_factorizeC');
    
    % Loads the co-occurrence data C, D1, and D2.        
    logger.info('+ Loading the data...');
    dataFile = sprintf('%s_train.mat', dataset);
    data = load(strcat(dataFolder, '/', dataFile));                
    logger.info('  - C file [%s] has been loaded!', dataFile);    
        
    % Compute the effective set of word indices.
    % (Note that words whose row sums are equal to 0 can be extseremly rare
    % but happen because we first tailor the vocabulary based on tf-idf, 
    % then compute the co-occurrence.)    
    I = setdiff(1:size(data.C, 1), find(sum(data.C, 2) == 0));
    C = data.C(I, I);
    D1 = data.D(I);
    D2 = data.D2(I, I);
    clear data;

    % Compute the row-sum and normalization.
    C_rowSums = sum(C, 2);
    Cbar = bsxfun(@rdivide, C, C_rowSums);      
    
    % For each rectification method,
    logger.info('+ Factorizing the data...');
    for i = 1:length(rectifiers)
        % Pick a rectifier.
        rectifier = rectifiers{i};
        logger.info('  - Rectifier = %s', rectifier);
          
        % Make a corresponding folder if it does not yet exist.
        outputFolder = sprintf('%s/%s_%s', outputBase, dataset, rectifier);
        if ~isfolder(outputFolder)
            mkdir(outputFolder);
        end
            
        % Load if the rectification result is already stored.
        rectFile = sprintf('%s/model_C-rect_K-%d.mat', outputFolder, K) ;
        if isfile(rectFile)
            load(rectFile, 'C_rect');            
            logger.info('  + Pre-rectified file is loaded!');
        else
            [C_rect, ~, elapsedTime] = rectification.rectifyC(C, K, rectifier);
            save(rectFile, 'C_rect', 'rectifier');
            logger.info('  + Finish the rectification! [%f]', elapsedTime);
        end        
        
        % For each optimization method,
        for j = 1:length(optimizers)
            % Pick an optimizer.
            optimizer = optimizers{j};    
            logger.info('    - Optimizer = %s', optimizer);
            
            % Make a corresponding subfolder if not exiting yet.
            outputSubFolder = sprintf('%s/%s', outputFolder, optimizer);
            if isfolder(outputSubFolder) == 0
                mkdir(outputSubFolder);
            end
            
            % Run the Rectified Anchor-Word Algorithm and store the resulting models.
            [S, B, A, Btilde, C_rectbar, C_rect_rowSums, ~, ~, ~, elapsedTime] = factorizeC(C_rect, K, 'skip', optimizer, dataset);     
            logger.info('    - Finish the factorization! [%f]', elapsedTime);             
            save(sprintf('%s/model_SBA_K-%d.mat', outputSubFolder, K), 'S', 'B', 'A', 'Btilde', 'optimizer');        
             
            % Generate the top words with respect to the type of data.
            inputDict = sprintf('%s/%s.dict', dataFolder, dataset);                
            resultBase = sprintf('%s/result_K-%d', outputSubFolder, K);
            if strncmp(dataset, 'movies', 6) == 1
                evaluation.generateTopMovies(S, B, 20, inputDict, I, resultBase);
            elseif strncmp(dataset, 'songs', 5) == 1
                evaluation.generateTopSongs(S, B, 20, inputDict, I, resultBase);
            else      
                evaluation.generateTopWords(S, B, 20, inputDict, I, resultBase);
            end

            % Save the evaluation results for various metrics.
            % Each experiment consists of two lines of results where the first line is against the original C 
            % and the second line is against the rectified C.
            outputFile1 = fopen(strcat(resultBase, '.metrics'), 'w');
            outputFile2 = fopen(strcat(resultBase, '.stdevs'), 'w');        
            [value1, stdev1] = evaluation.evaluateMetrics('all', S, B, A, Btilde, Cbar, C_rowSums, C, D1, D2, 1);                       
            [value2, stdev2] = evaluation.evaluateMetrics('all', S, B, A, Btilde, C_rectbar, C_rect_rowSums, C_rect, D1, D2);  
            fprintf(outputFile1, strcat(value1, '\n', value2, '\n'));
            fprintf(outputFile2, strcat(stdev1, '\n', stdev2, '\n'));        
            fclose(outputFile1);
            fclose(outputFile2);     
         end        
    end
    
    % Close the logger.
    logging.clearLogger('EXP_factorizeC_logger');
end




%%
% TODO:
%


