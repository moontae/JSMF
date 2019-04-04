%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR] = factorizeC_viaY(C, 100);
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values] = factorizeC_viaY(C, 100, 'AP');
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values] = factorizeC_viaY(C, 100, 'AP', 'activeSet');
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values] = factorizeC_viaY(C, 100, 'Baseline', 'expGrad', 'nips_N-5000');  
%


%%
% Main: factorizeC_viaY()
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
function [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values, elapsedTime] = factorizeC_viaY(C, K, rectifier, optimizer, dataset)    
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
        rectifier = 'ENN-randEig';
    end
    
    % Prepare a logger to record the result and performance.    
    logger = logging.getLogger('factorizeC_viaY_logger', 'path', sprintf('factorizeC_viaY%s_K-%d_%s_%s.log', dataset, K, rectifier, optimizer));
    logger.info('factorizeC_viaY');
    
    % Start the Rectified Anchor-Word Algorithm (RAWA).
    startTime = tic;
    
    % Rectify the co-occurrence matrix based on the option.
    logger.info('+ Start rectifying C...');
    [Y, ~, elapsedTime] = compression.rectifyC(C, K, rectifier);
    logger.info('  - Finish rectifying C! [%f]', elapsedTime);      
    
    % Run the compressed anchor-word algorithm.
    logger.info('+ Start factorizing C...');
    [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values, ~] = factorizeY(Y, K, optimizer, dataset);   
    
    % Finish the algorithm.
    elapsedTime = toc(startTime);
    logger.info('  - Finish factorizing C! [%f]', elapsedTime);
    
    % Close the logger.
    logging.clearLogger('factorizeC_viaY_logger');
end




%%
% TODO:
%
