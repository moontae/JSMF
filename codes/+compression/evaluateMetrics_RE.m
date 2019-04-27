%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Kun Dong
% Examples:
%   - [value, stdev] = evaluateMetrics_RE(S, Btilde, Y, E);
%   - [value, stdev] = evaluateMetrics_RE(S, Btilde, H);
%   


%%
% Main: evaluateMetrics_RE()
%
% Inputs:
%   - S: 1xK column vector having the basis indices
%   - Btilde: KxN cluster-object matrix where Btilde_{kn} = p(Z=k | X=n)
%   + Y_or_H: 
%     - Y: NxK rectified and compressed co-occurrence
%     - H: NxM document-word matrix
%   - E: NxN sparse correction matrix
%
% Outputs:
%   - value: recovery error (mean 2-norm of residuals)
%   - stdev: standard deviation of recovery error for all rows in Cbar
%
% Remarks: 
%   - This function approximates the recovery error in O(n) using low-rank structures.
%  
function [value, stdev] = evaluateMetrics_RE(S, Btilde, Y_or_H, E)
    % Compute the approximated recovery error based on the option.
    if nargin == 4
        % Given the compressed/rectified co-occurrence.
        [Cbarfun, CSbar] = recoveryError_Y(Y_or_H, E, S);
    elseif nargin == 3
        % Given the original word-document matrix.
        [Cbarfun, CSbar] = recoveryError_H(Y_or_H, S);
    end
    
    % Measure the subspace by anchor rows.
    N = size(Y_or_H, 1);
    [Q, R] = qr(CSbar, 0);
    [P, ~] = qr(randn(N, 100), 0);
    CQ = Cbarfun(Q);
    
    % Project into a random 100-D subspace.
    res = Cbarfun(P) - CQ * (Q' * P); 
    
    % Measure error in anchor rows subspace.
    error1 = sum(res.^2, 2) * (N / 100); 
    
    % Measure error in orthogonal complement.
    error2 = sum((CQ - (R * Btilde)').^2, 2); 
    
    % Return the final amounts of errors.
    error = sqrt(error1 + error2);    
    value = mean(error);
    stdev = std(error);
end


%%
% Inner: recoveryError_Y()
%
function [Cbarfun, CSbar] = recoveryError_Y(Y, E, S)
    C_rowSums = Y*sum(Y)' + sum(E, 2);
    Ybar = bsxfun(@rdivide, Y, C_rowSums);
    Ebar = bsxfun(@rdivide, E, C_rowSums);
    Cbarfun = @(x) Ybar * (Y' * x) + Ebar * x;
    CSbar = Y * Ybar(S, :)'+Ebar(:, S);
end


%%
% Inner: recoveryError_H()
%
function [Cbarfun, CSbar] = recoveryError_H(H, S)
    m = size(H, 2);
    f = sum(H);
    f = f .* (f-1);
    Hn = bsxfun(@rdivide, H, sqrt(f));
    dn = sum(bsxfun(@rdivide, H, f), 2);
    C_rowsums = (Hn * sum(Hn)' - dn) / m;
    Hnbar = bsxfun(@rdivide, Hn, C_rowsums);
    dnbar = dn ./ C_rowsums;
    Cbarfun = @(x) (Hnbar * (Hn' * x) - dnbar .* x) / m;
    CSbar = Hn*Hnbar(S, :)';
    CSbar(S, :) = CSbar(S, :)- diag(dnbar(S));
    CSbar = CSbar / m;
end




%%
% TODO:
%
