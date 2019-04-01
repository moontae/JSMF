%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%


%%
% Wrapper: rectifyC()
%
% Inputs:
%   - V: 
%   - D: 
%   - K: 
%   - rectifier:
%
% Outputs:
%   - Y:
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%
function [Y, E, elapsedTime] = rectifyC(C, K, rectifier)
    switch(rectifier)      
      case 'ENN-trunEig'
        % For Epsilon-NN method, E means a sparse correction.   
        [Y, E, elapsedTime] = compression.rectify_ENN(C, K, 'trunEig', 50);
      
      case 'ENN-randEig'
        % For Epsilon-NN method, E means a sparse correction.   
        [Y, E, elapsedTime] = compression.rectify_ENN(C, K, 'randEig', 50);      
        
      case 'PALM'
        % For PALM methods, E means a Y'.        
        [Y, E, ~, elapsedTime] = compression.rectify_PALM(C, K, 100, 1e-4);
      
      case 'IPALM'
        % For IPALM methods, E means a Y'.
        [Y, E, ~, elapsedTime] = compression.rectify_IPALM(C, K, 100, 1e-4);
            
      otherwise
        % No rectification, just truncated eigendecomposition.
        startTime = tic;
        Y = PSDk(@(x)C*x, size(C, 1), K);        
        E = [];
        elapsedTime = toc(startTime);
    end
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
    [U, e] = eigs(Cfun, N, K, 'LA', opt);
    
    % Return only the half of the result that can recover Rank-K projectdion by YY^T.
    Y = U*sqrt(max(e, 0));
end



%%
% TODO:
%
