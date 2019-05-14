%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Kun Dong
% Examples:
%   - [value, stdev] = evaluateClusters_AE(H, S, Btilde);
%   - [value, stdev] = evaluateClusters_AE(Y, E, S, Btilde);
%   


%%
% Main: evaluateClusters_AE()
%
% Inputs:
%   - B: NxK object-cluster matrix where B_{nk} = p(X=n | Z=k) 
%   - A: KxK cluster-cluster matrix where A_{kl} = p(Z1=k, Z2=l)
%   + Y_or_H: 
%     - Y: NxK rectified and compressed co-occurrence
%     - H: NxM document-word matrix
%   - E: NxN sparse correction matrix
%
% Outputs:
%   - value1: approximation error 
%   - value2: approximation error without considering diagonal elements
%
% Remarks: 
%   - This function approximates the approximation error in O(n) using low-rank
%     structures. The error of approximation < 2.5%.
%  
function [value1, value2] = evaluateClusters_AE(B, A, Y_or_H, E)
    % Compute the approximated approximation error based on the option.
    if nargin == 4
        % Given the compressed/rectified co-occurrence.        
        [Cfun, diagC] = approximationError_Y(Y_or_H, E);
    elseif nargin == 3
        % Given the original word-document matrix.
        [Cfun, diagC] = approximationError_H(Y_or_H);        
    end    
    
    % Measure the approximation error in full and off-diagonal entries.
    N = size(Y_or_H, 1);
    [P, ~] = qr(randn(N, 500), 0);
    BA = B * A;
    error1 = Cfun(P) - BA * (B' * P);
    value1 = norm(error1, 'fro') * sqrt(N / 500);
    diagError = norm(diagC - sum(BA .* B, 2));
    value2 = sqrt(value1 ^ 2 - diagError ^ 2);
end


%%
% Inner: approximationError_Y()
%
function [Cfun, diagC] = approximationError_Y(Y, E)
    Cfun = @(x) Y * (Y' * x) + E * x;
    diagC = sum(Y .^ 2, 2) + diag(E);
end


%%
% Inner: approximationError_H()
%
function [Cfun, diagC] = approximationError_H(H)
    M = size(H, 2);
    f = sum(H);
    f = f .* (f-1);
    Hn = bsxfun(@rdivide, H, sqrt(f));
    dn = sum(bsxfun(@rdivide, H, f), 2);
    Cfun = @(x) (Hn * (Hn' * x) - dn .* x) / M;
    diagC = (sum(Hn .^ 2, 2) - dn) / M;
end
