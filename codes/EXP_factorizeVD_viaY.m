%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Remark:
%   - This is a function that learns given number of topics and their 
%     correlations from a compressed training dataset ending with '_train_K-#.mat'.
%   - You can generate a Matlab standalone runtime executable for this
%     function by running experiments/EXP_factorizeY/EXP_factorizeY_make.m.
%   - Then you can submit a series of parallel jobs across differnet
%     datasets and topics by using a job queue in your cluster.
%   - Individual results must be later merged through the python script.
%
% Example: 
%   - EXP_factorizeVD_viaY('../dataset/real_mat', 'nips_N-5000', 5, '../models/real');
%   
function EXP_factorizeVD_viaY(input_folder, dataset, K, output_base)        
    % Setup the types of rectifications and optimizations.
    %rectifiers = {'Baseline', 'ENN'};
    rectifiers = {'ENN'};
    %optimizers = {'activeSet', 'admmDR', 'expGrad'};
    optimizers = {'activeSet'};
    
    % Setup the folder of dataset and the base folder to store the outputs.
    % (Note that the runtime executable will be stored on experiments folder
    % from which the relative path must be specified)
    dataFolder  = sprintf('%s/dataset/real_mat_comp', input_folder);
    modelFolder = sprintf('%s/models/real_comp', input_folder);    
    outputBase = output_base;
    
    % Prepare a logger to record the result and performance.
    logger = logging.getLogger('EXP_factorizeVD_viaY_logger', 'path', sprintf('EXP_factorizeVD_viaY_%s_K-%d.log', dataset, K));
    logger.info('EXP_factorizeVD_viaY');
    
    % Loads the compressed co-occurrence data V and D
    logger.info('+ Loading the compressed data...');
    dataFile = sprintf('%s_train_K-%d.mat', dataset, K);
    load(strcat(dataFolder, '/', dataFile), 'V', 'D');                
    logger.info('  - V and D file [%s] has been loaded!', dataFile);         
    
    % For each rectification method,
    logger.info('+ Factorizing the compressed data...');
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
        rectFile = sprintf('%s/model_YE_K-%d.mat', modelFolder, K) ;
        if isfile(rectFile)
            load(rectFile, 'Y');            
            logger.info('  + Pre-rectified file is loaded!');
        else
            [Y, E, elapsedTime] = compression.rectifyVD(V, D, K, rectifier);
            save(sprintf('%s/model_YE_K-%d.mat', outputFolder, K), 'Y', 'E', 'rectifier');
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
            [S, B, A, Btilde, C_rectbar, C_rect_rowSums, ~, ~, ~, elapsedTime] = factorizeY(Y, K, optimizer, dataset);   
            logger.info('    - Finish the factorization! [%f]', elapsedTime);             
            save(sprintf('%s/model_SBA_K-%d.mat', outputSubFolder, K), 'S', 'B', 'A', 'Btilde', 'optimizer');                                
             
            % Decide the non-zero indices.
            I = setdiff(1:size(Y, 1), find(C_rect_rowSums == 0));    
            
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
            [value1, stdev1] = evaluation.evaluateMetrics('allForComp', S, B, A, Btilde, C_rectbar, C_rect_rowSums, 1);                       
            [value2, stdev2] = evaluation.evaluateMetrics('allForComp', S, B, A, Btilde, C_rectbar, C_rect_rowSums);  
            fprintf(outputFile1, strcat(value1, '\n', value2, '\n'));
            fprintf(outputFile2, strcat(stdev1, '\n', stdev2, '\n'));        
            fclose(outputFile1);
            fclose(outputFile2);     
         end        
    end
    
    % Close the logger.
    logging.clearLogger('EXP_factorizeVD_viaY_logger');
end
    
    
    
    
    
