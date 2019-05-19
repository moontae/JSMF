%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [value, stdev] = evaluateClusters_TS(B);
%   - [value, stdev] = evaluateClusters_TS(B, gamma);
%   


%%
% Main: evaluateClusters_TS()
%
% Inputs:
%   - B: NxK object-cluster matrix where B_{nk} = p(X=n | Z=k) 
%   - option: either gamma-2 or gamma-star based on the notion of separability
%   - S: 1xK column vector having the basis indices
%
% Outputs:
%   - value: the average separability across different clusters.

% Remarks: 
%   - This function measures two different relaxed notion of p-separability
%     so called gamma-2 and gamma-star separability based on the clusters.
%   - If the optional argument S is given, we assume the highest separable
%     rows of B are automatically S without actual calculations.
%  
function [value, stdev] = evaluateClusters_TS(B, option, S)
    [N, K] = size(B);
    
    % Find the first maximum entry in each row.
    [L1, I1] = max(B, [], 2);        
    
    % Measrue gamma-2 separability.
    if strcmp(option, 'gamma-2')
        % Change those entries to negative infinities.
        B(sub2ind(size(B), 1:N, I1')) = -Inf;
        
        % Find the second maximum entry in each row.
        % Note that if we can suppress the second largest entry in each row,
        % all other values are automatically suppressed.
        [L, ~] = max(B, [], 2);        
    else
        % Change those entries to zeros.
        B(sub2ind(size(B), 1:N, I1')) = 0;
                
        % Compute the sum of all non-maximum entries for every row.
        % Note that suppressing all other entries is the slightly stonger
        % and softer notion of separability than gamma-2 version.
        L = sum(B, 2);
    end
    
    % Compute the smallest gamma values that can suppress either the second
    % largest entry or sum of all non-maximum entries by gamma*L1 per rows.
    gammas = L ./ L1;
        
    % If anchor objects are given,
    if nargin > 2
        % Pick the subset of gammas corresponding to the anchor objects.
        gammas = gammas(S);
    else
        % Find the K smallest ratios from existing gammas.
        [~, I2] = sort(gammas, 'ascend');
        gammas = gammas(I2(1:K));
    end
    
   % Return the mean and standard deviations.
   value = mean(gammas);
   stdev = std(gammas);
end

        
        
        
        
    



