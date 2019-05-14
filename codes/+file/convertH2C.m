%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [C, C_rowSums] = file.convertH2C(H)


%%
% Main: convertH2C()
%
% Remarks:
%   - This function converts the word-document matrix H into the 
%     co-occurrence and its row-wise summation information.
%   - Even if H consists of large vocabulary, C_rowSums can be efficiently
%     computed, being useful for compression-based factorization.
%
function [C_rowSums, C] = convertH2C(H)
    [N, M] = size(H);

    % Create Hhat whose m-th column is h_m / sqrt(n_m*(n_m - 1)*M).
    H_colSums = sum(H, 1);
    H_denominators = H_colSums.*(H_colSums - 1) * M;
    Hhat = bsxfun(@rdivide, H, sqrt(H_denominators));

    % Create Hdiag which is sum of diag(h_m) / (n_m*(n_m - 1)*M) across all examples.
    Hdiag = diag(sum(bsxfun(@rdivide, H, H_denominators), 2));
    
    % Compute based on the output options.
    if nargout < 2        
        % If C_rowSums is only necessary,        
        C_rowSums = full(Hhat*(Hhat'*ones(N, 1)) - Hdiag*ones(N, 1));
    else
        % If C is also neessary,
        C = full(Hhat*Hhat' - Hdiag);
        C_rowSums = sum(C, 2);
    end
end



