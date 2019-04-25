%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - A = recoverA(C, B, S);
%


%%
% Main: recoverA()
%
% Inputs:
%   - C: NxN original co-occurrence matrix
%   - B: NxK object-cluster matrix (column-stochastic)
%   - S: 1xK vector having the row indices of approximate basis vectors
%   + option: the method of recovery (default = 'diagonal')
%     - diagonal:
%     - pseudoInverse:
%     - optimize:
%
% Outputs:
%   - A: KxK cluster-cluser matrix where A_{kl} = p(Z1=k | Z2=l)
%   - elapsedTime: total elapsed amount of seconds
%
% Remarks:
%   - This function recovers the matrix B by two different methods.
%
function [A, elapsedTime] = recoverA(C, B, S, option)
    % Set the default option.
    if nargin < 4
        option = 'diagonal';
    end

    % Print out the initial status.
    fprintf('[inference.recoverA] Start recovering the cluster-cluster A...\n');
    
    startTime = tic;
    switch option
      case 'diagonal' 
        A = diagonalRecovery(C, B, S);
      case 'pseudoInverse'
        A = pseudoInverseRecovery(C, B);
      case 'optimize'
        A = optimizeRecovery(C, B, S);  
      otherwise
        error('  * Underfined option [%s] is given!\n', option);                  
    end
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish recovering A!\n');
    fprintf('  - [%s] recovery is used.\n', option);
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);     
end


%%
% Inner: diagonalRecovery()
%
% Inputs:
%   - C: NxN original co-occurrence matrix
%   - B: NxK object-cluster matrix (column-stochastic)
%   - S: 1xK vector having the row indices of approximate basis vectors
%
% Outputs:
%   - A: KxK cluster-cluser matrix where A_{kl} = p(Z1=k | Z2=l)
%
% Remarks:
%   - This function recovers the matrix A by diagonal methods which
%     maximally utilized the separability assumption.
%   - Note that diagonal recovery guarantees that the result A becomes 
%     doubly nonnegative.
%
function A = diagonalRecovery(C, B, S)
    C_SS = C(S, S);
    B_S = B(S, :);
    invB_S = diag(1.0 ./ diag(B_S));
    A = invB_S*C_SS*invB_S;
end


%%
% Inner: pseudoInverseRecovery()
%
% Inputs:
%   - C: NxN original co-occurrence matrix
%   - B: NxK object-cluster matrix (column-stochastic)
%
% Outputs:
%   - A: KxK cluster-cluser matrix where A_{kl} = p(Z1=k | Z2=l)
%
% Remarks:
%   - This function recovers the matrix A by multyplying pseudo-inverse.
%   - Note that pseudo-inverse in the orignal paper has many non-negligible
%     negative entries, being failed in become a joint distribution
%
function A = pseudoInverseRecovery(C, B)
    pinvB = pinv(B);
    A = pinvB*C*pinvB';
end


%%
% Inner: optimizeRecovery()
%
function A = optimizeRecovery(C, B, S, T)
    if nargin < 4
        T = 100;
    end
    
    % Perform the diagonal recovery for initialization.
    C_SS = C(S, S);
    B_S = B(S, :);
    invB_S = diag(1.0 ./ diag(B_S));
    A = invB_S*C_SS*invB_S;
    
    % Perform further multiplicative update to minimize approximation error.
    BtCB = B'*C*B;
    BtB = B'*B;
    for t = 1:int32(T)
        A_prev = A;
        A = A .* ((BtCB) ./ (BtB*A*BtB));
        A = A ./ sum(sum(A));
        
        diff = norm(A - A_prev, 'fro');        
        if diff < 0.0001
            fprintf('  - Converged at %d-th iteration!\n', t);
            break
        end        
    end
end




%%
% TODO:
%