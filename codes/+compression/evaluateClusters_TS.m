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
%   - gamma: proportion to suppress plain objects by the mass of the novel object
%
% Outputs:
%   - value: the average separability across different topics

% Remarks: 
%   - This function computes either the soft-separability or hard version
%     in terms of gamma-approximate separability.
%  
function [value, I] = evaluateClusters_TS(B, option)
    % Soft separability.
    if nargin < 2
        % Perform row-normalization of the clsuter matrix.
        B_rowSums = sum(B, 2);
        Bbar = bsxfun(@rdivide, B, B_rowSums);  
        [I, value] = munkres(1 - Bbar);
    end
end



