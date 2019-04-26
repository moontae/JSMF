%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Kun Dong
% Examples:
%   - [value, stdev] = reconstructionError_lr(H, S, Btilde);
%   - [value, stdev] = reconstructionError_lr(Y, E, S, Btilde);
%   


%%
% Main: recoveryError_lr()
%
% Inputs:
%   - H: NxM document-word matrix
%   - Y: NxK rectified and compressed co-occurrence
%   - B: NxK object-cluster matrix where B_{nk} = p(X=n | Z=k) 
%   - A: KxK cluster-cluster matrix where A_{kl} = p(Z1=k, Z2=l)
%
% Outputs:
%   - value1: Reconstruction error
%   - value2: Reconstruction error excluding diagonal elements
%
% Remarks: 
%   - This function approximates the reconstruction error in O(n) using low-rank
%     structures. Approximation error < 2.5%.
%  
function [value1, value2] = reconstructionError_lr(Y, E, B, A)
    if nargin == 4
        [Cfun, diagC] = reconstructionError_Y(Y, E);
    elseif nargin == 3
        [Cfun, diagC] = reconstructionError_H(Y);
        A = B;
        B = E;
    end
    n = size(Y, 1);
    [P, ~] = qr(randn(n, 500), 0);
    BA = B * A;
    error1 = Cfun(P) - BA * (B' * P);
    value1 = norm(error1, 'fro') * sqrt(n / 500);
    diagError = norm(diagC - sum(BA .* B, 2));
    value2 = sqrt(value1 ^ 2 - diagError ^ 2);
end

function [Cfun, diagC] = reconstructionError_H(H)
    m = size(H, 2);
    f = sum(H);
    f = f .* (f-1);
    Hn = bsxfun(@rdivide, H, sqrt(f));
    dn = sum(bsxfun(@rdivide, H, f), 2);
    Cfun = @(x) (Hn * (Hn' * x) - dn .* x) / m;
    diagC = (sum(Hn .^ 2, 2) - dn) / m;
end

function [Cfun, diagC] = reconstructionError_Y(Y, E)
    Cfun = @(x) Y * (Y' * x) + E * x;
    diagC = sum(Y .^ 2, 2) + diag(E);
end