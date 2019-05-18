%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Kun Dong & Moontae Lee
% Examples:
%


%%
% Main: rectify_ENN() 
%
% Inputs:
%   - C: NxN original (joint-stochastic) co-occurrence matrix 
%   - K: the number of basis vectors
%   + option: choose how to initialize
%     - trunEig: use truncated eigendecomposition
%     - randEig: use randomized eigendecomposition
%     - otherwise: use the user-specified V and D from arguments
%   - T: the number of iterations.
%   - V: user-specified matrix of eigenvectors
%   - D: user-specified matrix of eigenvalues
%
% Outputs:
%   - Y: low-rank compression for rectified C
%   - E: sparse correction for non-negative entries of the final outcome
%
% Remarks: 
%   - There are a few parameters that could be set as inputs.
%  
function [Y, E, elapsedTime] = rectify_ENN(C, K, option, T, V, D)
    % Set the default number of iterations.
    if nargin < 4
        T = 50;
    end

    % Set the default option.
    if nargin < 3
        option = 'randEig';
    end
            
    % Print out the initial status.
    fprintf('[compression.rectify_ENN] Start compressing + rectifying by Epsilon-NN...\n');
    fprintf('- Option = %s\n', option);
        
    % Find the initial Y that can approximate C by YY^T.
    startTime = tic;
    switch(option)
      case 'trunEig'    
        % Perform an explicit truncated eigenvalue decomposition.
        Y = PSDk(@(x)C*x, size(C, 1), K);
      
      case 'randEig'
        % Create a random probe matrix by using 2K columns.
        P = randn(size(C, 1), K+K);
        
        % Perform a randomized eigendecomposition. While it still uses C as the matrix,
        % randEig only applies C as an operator. Thus C needs not to be explicit anymore.
        % (Note that we uses 2 steps of power iteration refinement, which is necessary)        
        [V, D] = compression.randEig(P, C, K, 50);
        Y = V.*sqrt(max(diag(D)', 0));
      
      otherwise
        % Use directly input arguments.
        Y = V.*sqrt(max(diag(D)', 0));
    end    
    
    % Decide the number of rows to be classified as large norms.
    % For dataset with small vocabulary, 5K*log(K) might be bigger than the
    % size of vocabulary if K is very large.
    l = ceil(min(size(Y, 1), 5*K*log(K)));
    
    % Evaluate the square of two-norm per each row and sorts decreasingly.
    [~, I] = sort(sum(Y.^2, 2), 'descend');
        
    % For each iteration in epsilon-NN rectification,
    N = size(Y, 1);
    for t = 1:int32(T)
        % Find the negative entries inside the large norm area.
        % Say smaller norm area = section (2, 2)
        checkNN = min(Y*Y(I(1:l), :)', 0);
        [rows, cols, values] = find(checkNN);
        
        % Create a correction for non-negative entries in this area.
        % E(rows[i], I(cols[j])) = -values[j]) as a sparse non-negative matrix.
        % E currently covers both sections (1, 1) and (1, 2).
        E = sparse(rows, I(cols), -values, N, N);
        
        % Make cor_NN cover upto section (2, 1), which is just symmetric to section (1, 2).
        E = max(E, E');
        
        % Compute the weight correction for making it to be a normalized by adding the matrix: r*ee^T.
        % 1 - sum of all entries in (YY^T + cor_NN) = 1 - sum of all entries in YY^T - sum of all entries in cor_NN
        % e^T(YY^T)e = (Y^T e)^T (Y^T e) = sum(sum(Y, 1).^2).        
        r = (1 - sum(sum(Y, 1).^2) - sum(E(:)))/(N^2);
        
        % Apply three orthogonal projection steps at once.
        Y = PSDk(@(x)(Y*(Y'*x) + E*x + r*sum(x)*ones(N,1)), N, K);
        
        % Evaluate the square of two-norm per each row and finds the large norm area again.
        [~, I] = sort(sum(Y.^2, 2), 'descend');
    end
    elapsedTime = toc(startTime);
    
    % Get a sparse correction for non-negative entries of the final outcome.
    % Note that YY^T + E can still have a small negative entries.
    checkNN = min(Y*Y(I(1:l), :)', 0);
    [rows, cols, values] = find(checkNN);
    E = sparse(rows, I(cols), -values, N, N);
    E = max(E, E');    
    
     % Print out the final status.
    fprintf('+ Finish Epsilon-NN!\n');    
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);     
end


%% 
% Inner: PSDk()
%
% Remark:
%   - Projects to Rank-K Positive SemiDefinite matrix
%
function Y = PSDk(Cfun, N, K)
    % Decide options about the matrix.
    opt.issym = 1;
    opt.isreal = 1;
    
    % Find just the first K eigenvalues and eigenvectors.
    [U, Lambda] = eigs(Cfun, N, K, 'LA', opt);
    
    % Return only the half of the result that can recover rank K projection by YY^T.
    Y = U*sqrt(max(Lambda, 0));
end




%%
% TODO:
%


