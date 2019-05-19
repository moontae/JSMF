%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Remark:
%   - This is a function that learns given number of topics and their 
%     correlations from a compressed dataset ending with '_K-#.mat'.
%   - You can generate a Matlab standalone runtime executable for this
%     function by running experiments/EXP_factorizeY/EXP_factorizeY_make.m.
%   - Then you can submit a series of parallel jobs across differnet
%     datasets and topics by using a job queue in your cluster.
%   - Individual results must be later merged through the python script.
%
% Example: 
%   - EXP_factorizeVD_viaY('../../jsmf-dataset', 'nytimes_N-60000', 5, '../models/real');
%   - EXP_factorizeVD_viaY('../../jsmf-dataset', 'nips_N-5000', 5, '../models/real');
%   
function EXP_factorizeVD_viaY(input_folder, dataset, K, output_base)        
    % Setup the types of rectifications and optimizations.
    rectifiers = {'ENN'};
    %rectifiers = {'ENN'};
    %optimizers = {'activeSet', 'admmDR', 'expGrad'};
    optimizers = {'activeSet'};
    
    % Decide the ratio for computing partial recovery error.    
    N_base = 0;
    if strncmp(dataset, 'nips', 4) == 1
        N_base = 1250;
    elseif strncmp(dataset, 'nytimes', 7) == 1
        N_base = 7500;
    elseif strncmp(dataset, 'movies', 6) == 1
        N_base = 1250;
    elseif strncmp(dataset, 'songs', 5) == 1
        N_base = 5000;
    elseif strncmp(dataset, 'blog', 4) == 1
        N_base = 500;
    elseif strncmp(dataset, 'yelp', 4) == 1
        N_base = 200;
    end
    leftVocabIndex = strfind(dataset, 'N-') + 2;    
    N = str2double(dataset(leftVocabIndex:end));
    ratio = N_base/N;
        
    % Setup the folder of dataset and the base folder to store the outputs.
    % (Note that the runtime executable will be stored on experiments folder
    % from which the relative path must be specified)
    dataFolder  = sprintf('%s/dataset/real_mat_comp', input_folder);
    modelFolder = sprintf('%s/models/real_comp', input_folder);    
    outputBase = output_base;
    
    % Prepare a logger to record the result and performance.
    logger = logging.getLogger('EXP_factorizeVD_viaY_logger', 'path', sprintf('EXP_factorizeVD_viaY_%s_K-%d.log', dataset, K));
    logger.info('EXP_factorizeVD_viaY');
    
    % Load the compressed co-occurrence data (V, D) and the original data H.
    logger.info('+ Loading the compressed data...');
    dataFilename = sprintf('%s_K-%d.mat', dataset, K);
    load(strcat(dataFolder, '/', dataFilename), 'V', 'D', 'H');                
    logger.info('  - (V, D) and H file [%s] has been loaded!', dataFilename);             
    logger.info('  - Current ratio = %.2f', ratio);             
    
    % Compute the row summation of the original C matrix.
    % Note that the row summation of the rectified C will be recovered later from the factorizeY.
    C_rowSums = file.convertH2C(H);
    
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
        rectFilename = sprintf('%s/model_YE_K-%d.mat', modelFolder, K) ;
        if isfile(rectFilename)
            load(rectFilename, 'Y');            
            logger.info('  + Pre-rectified file is loaded!');
        else
            [Y, E, rectifyTime] = compression.rectifyVD(V, D, K, rectifier);
            save(sprintf('%s/model_YE_K-%d.mat', outputFolder, K), 'Y', 'E', 'rectifier');
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
            [S, B, A, Btilde, Ebar, C_rect_rowSums, ~, E, ~, factorizeTime] = factorizeY(Y, K, optimizer, dataset);   
            logger.info('    - Finish the factorization! [%f]', factorizeTime);             
            save(sprintf('%s/model_SBA_K-%d.mat', outputSubFolder, K), 'S', 'B', 'A', 'Btilde', 'E', 'optimizer');                                
             
            % Decide the non-zero indices.
            I = setdiff(1:size(Y, 1), find(C_rect_rowSums == 0));    
            
            % Generate the top words with respect to the type of data.
            dictFilename = sprintf('%s/%s.dict', dataFolder, dataset);                
            resultBase = sprintf('%s/result_K-%d', outputSubFolder, K);
            if strncmp(dataset, 'movies', 6) == 1
                evaluation.generateTopMovies(S, B, 20, dictFilename, I, resultBase);
            elseif strncmp(dataset, 'songs', 5) == 1
                evaluation.generateTopSongs(S, B, 20, dictFilename, I, resultBase);
            else      
                evaluation.generateTopWords(S, B, 20, dictFilename, I, resultBase);
            end           
            
            % Save the evaluation results for various metrics.
            % Each experiment consists of two lines of results where the first line is against the original C 
            % and the second line is against the rectified C.
            [metric1, stdev1] = compression.evaluateClusters('allGivenH', S, B, A, Btilde, H, C_rowSums);                       
            [metric2, stdev2] = compression.evaluateClusters('allGivenYE', S, B, A, Btilde, Y, E, C_rect_rowSums, ratio);                              
            metricTitle = compression.evaluateClusters('allGivenH', [], [], [], [], [], [], -1);
            
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
    logging.clearLogger('EXP_factorizeVD_viaY_logger');
end
    
    
    
    
    
