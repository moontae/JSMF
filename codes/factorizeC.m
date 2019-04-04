%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR] = factorizeC(C, 100);
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values] = factorizeC(C, 100, 'AP');
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values] = factorizeC(C, 100, 'AP', 'activeSet');
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values] = factorizeC(C, 100, 'Baseline', 'expGrad', 'nips_N-5000');  
%


%%
% Main: factorizeC()
%
% Inputs:
%   - C: NxN original co-occurrence matrix (must be joint-stochastic)
%   - K: the number of basis vectors (i.e., the number of topics)
%   + rectifier:
%     - 'DC': Diagonal Completion (only low-rank)
%     - 'AP': Alternating Projection (main method)
%     - 'DP': Dykstra Projection (experimental)
%     - otherwise: no rectification (equivalent to the vanilla Anchor Word Algorithm)
%
% Outputs:
%   - S:           1xK column vector having the basis indices
%   - B:           NxK object-cluster matrix where B_{nk} = p(X=n | Z=k) 
%   - A:           KxK cluster-cluster matrix where A_{kl} = p(Z1=k, Z2=l)
%   - Btilde:      KxN cluster-object matrix where Btilde_{kn} = p(Z=k | X=n) 
%   - Cbar:        NxN row-normalized co-occurrence matrix where Cbar_{ij} = p(X2=j | X1=i)
%   - C_rowSums:   Nx1 vector indicating the row-wise sum of C where C_rowSums_i = p(X=i)
%   - C:           NxN updated C matrix after the rectification step
%   - values:      The outputs from the rectification step
%   - diagR:       1xK vector indicating the scores of each basis vector
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%   - This function performs the overall joint-stochastic matrix factorization (a.k.a Rectified Anchor-Words Algorithm (RAWA).
%   - Run the rectification first if specified.
%   - Run the anchor-word algorithm on the rectified co-occurrence matrix.
%  
function [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values, elapsedTime] = factorizeC(C, K, rectifier, optimizer, dataset)    
    % Set the default parameters.
    if nargin < 5
        dataset = '';
    else
        dataset = sprintf('_%s', dataset);
    end    
    if nargin < 4
        optimizer = 'activeSet';
    end
    if nargin < 3
        rectifier = 'AP';
    end
    
    % Prepare a logger to record the result and performance.    
    logger = logging.getLogger('factorizeC_logger', 'path', sprintf('factorizeC%s_K-%d_%s_%s.log', dataset, K, rectifier, optimizer));
    logger.info('factorizeC');
    
    % Start the Rectified Anchor-Word Algorithm (RAWA).
    startTime = tic;
    
    % Step 0: Rectify the co-occurrence matrix based on the option.
    logger.info('+ Start rectifying C...');
    [C, values, elapsedTime] = rectification.rectifyC(C, K, rectifier);
    logger.info('  - Finish rectifying C! [%f]', elapsedTime);
      
    % Perform row-normalization for the (rectified) co-occurrence matrix C.    
    C_rowSums = sum(C, 2);
    Cbar = bsxfun(@rdivide, C, C_rowSums);  
    
    % Step 1: Find the given number of bases S. (i.e., set of indices correspondingtStart to the anchor words)
    logger.info('+ Start finding the set of anchor bases S...');
    [S, diagR, elapsedTime] = inference.findS(Cbar, K);
    logger.info('  - Finish finding S! [%f]', elapsedTime);    

    % Step 2: Recover object-cluster matrix B. (i.e., recovers word-topic matrix)
    logger.info('+ Start recovering the object-cluster B...');
    [B, Btilde, elapsedTime] = inference.recoverB(Cbar, C_rowSums, S, optimizer);
    logger.info('  - Finish recovering B! [%f]', elapsedTime);
    
    % Step 3: Recover cluster-cluster matrix A. (i.e., recovers topic-topic matrix)
    logger.info('+ Start recovering the cluster-cluster A...');
    [A, elapsedTime] = inference.recoverA(C, B, S);
    logger.info('  - Finish recovering A! [%f]', elapsedTime);
    
    % Finish the algorithm.
    elapsedTime = toc(startTime);
    logger.info('- Finish factorizing C! [%f]', elapsedTime);
    
    % Close the logger.
    logging.clearLogger('factorizeC_logger');
end




%%
% TODO:
%
