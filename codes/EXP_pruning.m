%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee & Sungjun Cho
% Remark:
%   - This is a function that learns given number of topics and their 
%     correlations from a training dataset ending with '.mat'.
%   - You can generate a Matlab standalone runtime executable for this
%     function by running experiments/EXP_factorizeC/EXP_factorizeC_make.m.
%   - Then you can submit a series of parallel jobs across differnet
%     datasets and topics by using a job queue in your cluster.
%   - Individual results must be later merged through the python script.
%   - P denotes the pruning parameter: using P = 0.50 means that we prune
%     out all words that appear in more than half of the documents.
%
% Example: 
%   - EXP_pruning('../../jsmf-dataset', 'nips_N-5000', 0.50, 5, '../models/pruning');
%
function EXP_pruning(input_folder, dataset, p, K, output_base)        
    % Setup the types of rectifications and optimizations.
    %rectifiers = {'Baseline', 'AP', 'DC', 'DP'};    
    rectifiers = {'AP'};
    %optimizers = {'activeSet', 'admmDR', 'expGrad'};
    optimizers = {'activeSet'};
        
    % Setup the folder of dataset and the base folder to store the outputs.
    % (Note that the runtime executable will be stored on experiments folder
    % from which the relative path must be specified)
    dataFolder  = sprintf('%s/dataset/real_mat_pruning', input_folder);
    modelFolder = sprintf('%s/models/real_pruning', input_folder);    
    outputBase = output_base;
    P = round(p*100);
    
    % Prepare a logger to record the result and performance.
    logger = logging.getLogger('EXP_pruning_logger', 'path', sprintf('EXP_pruning_%s_P-%d_K-%d.log', dataset, P, K));
    logger.info('EXP_pruning');
    
    % Loads the co-occurrence data C, D1, and D2.        
    logger.info('+ Loading the data...');
    dataFilename = sprintf('%s_P-%d.mat', dataset, P);
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
        rectFilename = sprintf('%s/model_C-rect_P-%d_K-%d.mat', modelFolder, P, K) ;
        if isfile(rectFilename)
            load(rectFilename, 'C_rect');            
            logger.info('  + Pre-rectified file is loaded!');
        else
            [C_rect, values, rectifyTime] = rectification.rectifyC(C, K, rectifier);
            save(sprintf('%s/model_C-rect_P-%d_K-%d.mat', outputFolder, P, K), 'C_rect', 'values', 'rectifier');
            logger.info('  + Finish the rectification! [%f]', rectifyTime);
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
            [S, B, A, Btilde, C_rectbar, C_rect_rowSums, ~, ~, ~, factorizeTime] = factorizeC(C_rect, K, 'skip', optimizer, dataset);     
            logger.info('    - Finish the factorization! [%f]', factorizeTime);             
            save(sprintf('%s/model_SBA_P-%d_K-%d.mat', outputSubFolder, P, K), 'S', 'B', 'A', 'Btilde', 'optimizer');        
             
            % Generate the top words with respect to the type of data.
            dictFilename = sprintf('%s/%s_P-%d.dict', dataFolder, dataset, P);  
            resultBase = sprintf('%s/result_P-%d_K-%d', outputSubFolder, P, K);
            if strncmp(dataset, 'movies', 6) == 1
                evaluation.generateTopMovies(S, B, 20, dictFilename, I, resultBase);
            elseif strncmp(dataset, 'songs', 5) == 1
                evaluation.generateTopSongs(S, B, 20, dictFilename, I, resultBase);
            else      
                evaluation.generateTopWords(S, B, 20, dictFilename, I, resultBase);
            end

            % Evaluate the metrics against both original and rectified C.
            [metric1, stdev1] = evaluation.evaluateClusters('all', S, B, A, Btilde, Cbar, C_rowSums, C, D1, D2);                       
            [metric2, stdev2] = evaluation.evaluateClusters('all', S, B, A, Btilde, C_rectbar, C_rect_rowSums, C_rect, D1, D2);  
            metricTitle = evaluation.evaluateClusters('all', [], [], [], [], [], [], [], [], [], -1);
            
            % Save the evaluation results for metric values.
            metricFile = fopen(strcat(resultBase, '.metrics'), 'w');                        
            fprintf(metricFile, sprintf('%s %14s %14s\n', metricTitle, 'RectifyTime', 'FactorizeTime'));
            fprintf(metricFile, sprintf('%s %14.3f %14.3f\n', metric1, rectifyTime, factorizeTime));
            fprintf(metricFile, sprintf('%s %14.3f %14.3f\n', metric2, rectifyTime, factorizeTime));            
            fclose(metricFile);            
            
            % Save the evaluation result for standard deviation values.
            stdevFile = fopen(strcat(resultBase, '.stdevs'), 'w');                  
            fprintf(stdevFile, sprintf('%s %14s %14s\n', metricTitle, 'RectifyTime', 'FactorizeTime'));
            fprintf(stdevFile, sprintf('%s %14.3f %14.3f\n', stdev1, rectifyTime, factorizeTime));
            fprintf(stdevFile, sprintf('%s %14.3f %14.3f\n', stdev2, rectifyTime, factorizeTime));            
            fclose(stdevFile);       
         end        
    end
    
    % Close the logger.
    logging.clearLogger('EXP_pruning_logger');
end