%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
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
%   - EXP_factorizeC_viaY('../dataset/real_mat', 'nips_N-5000', 5, '../models/real');
%   
function EXP_factorizeC_viaY(input_folder, dataset, K, output_base)        
    % Setup the types of rectifications and optimizations.
    rectifiers = {'NONE-trunEig', 'ENN-trunEig', 'ENN-randEig', 'PALM', 'IPALM'};
    %rectifiers = {'ENN-trunEig', 'ENN-randEig', 'PALM', 'IPALM'};
    %optimizers = {'activeSet', 'admmDR', 'expGrad'};
    optimizers = {'activeSet'};
    
    % Setup the folder of dataset and the base folder to store the outputs.
    % (Note that the runtime executable will be stored on experiments folder
    % from which the relative path must be specified)
    dataFolder  = sprintf('%s/dataset/real_mat', input_folder);
    modelFolder = sprintf('%s/models/real', input_folder);    
    outputBase = output_base; 
    
    % Prepare a logger to record the result and performance.
    logger = logging.getLogger('EXP_factorizeC_viaY_logger', 'path', sprintf('EXP_factorizeC_viaY_%s_K-%d.log', dataset, K));
    logger.info('EXP_factorizeC_viaY');
    
    % Loads the co-occurrence data C, D1, and D2.        
    logger.info('+ Loading the data...');
    dataFilename = sprintf('%s_train.mat', dataset);
    load(strcat(dataFolder, '/', dataFilename));                
    logger.info('  - C file [%s] has been loaded!', dataFilename);    
    
    % Compute the effective set of word indices.
    % (Note that words whose row sums are equal to 0 can be extseremly rare
    % but happen because we first tailor the vocabulary based on tf-idf, 
    % then compute the co-occurrence.)    
    I = setdiff(1:size(C, 1), find(sum(C, 2) == 0));
    C = C(I, I);
    D1 = D1(I);
    D2 = D2(I, I);    
    
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
        rectFilename = sprintf('%s/model_YE_K-%d.mat', modelFolder, K) ;
        if isfile(rectFilename)
            load(rectFilename, 'Y');            
            logger.info('  + Pre-rectified file is loaded!');
        else
            [Y, E, elapsedTime] = compression.rectifyC(C, K, rectifier);
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
            [S, B, A, Btilde, C_rectbar, C_rect_rowSums, ~, E, ~, elapsedTime] = factorizeY(Y, K, optimizer, dataset);   
            logger.info('    - Finish the factorization! [%f]', elapsedTime);             
            save(sprintf('%s/model_SBA_K-%d.mat', outputSubFolder, K), 'S', 'B', 'A', 'Btilde', 'E', 'optimizer');                                
                         
            % Generate the top words with respect to the type of data.
            dicdtFilename = sprintf('%s/%s.dict', dataFolder, dataset);                
            resultBase = sprintf('%s/result_K-%d', outputSubFolder, K);
            if strncmp(dataset, 'movies', 6) == 1
                evaluation.generateTopMovies(S, B, 20, dicdtFilename, I, resultBase);
            elseif strncmp(dataset, 'songs', 5) == 1
                evaluation.generateTopSongs(S, B, 20, dicdtFilename, I, resultBase);
            else      
                evaluation.generateTopWords(S, B, 20, dicdtFilename, I, resultBase);
            end

            % Save the evaluation results for various metrics.
            % Each experiment consists of two lines of results where the first line is against the original C 
            % and the second line is against the rectified C.
            outputFile1 = fopen(strcat(resultBase, '.metrics'), 'w');
            outputFile2 = fopen(strcat(resultBase, '.stdevs'), 'w');             
            [value1, stdev1] = evaluation.evaluateMetrics('all', S, B, A, Btilde, Cbar, C_rowSums, C, D1, D2, 1);                       
            [value2, stdev2] = evaluation.evaluateMetrics('all', S, B, A, Btilde, C_rectbar, C_rect_rowSums, Y*Y' + E, D1, D2); 
            fprintf(outputFile1, strcat(value1, '\n', value2, '\n'));
            fprintf(outputFile2, strcat(stdev1, '\n', stdev2, '\n'));        
            fclose(outputFile1);
            fclose(outputFile2);     
         end        
    end
    
    % Close the logger.
    logging.clearLogger('EXP_factorizeC_viaY_logger');
end
    
    
    
    
    
