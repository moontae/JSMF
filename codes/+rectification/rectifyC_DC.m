%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee & David Bindel
% Modified: April, 2019
% Examples:
%   - [C_rect, values] = rectifyC_DC(C, 100);
%   - [C_rect, values] = rectifyC_DC(C, 100, 5);
%


%%
% Main: rectifyC_DC()
%
% Inputs:
%   - C: NxN co-occurrence matrix (joint-stochastic)
%   - K: the low-rank to enforce
%   - T: the number of random permutations
%
% Outputs:
%   - C: NxN co-occurrence matrix (low-rank)
%   - values: TxN matrix where each row is the completed diagonal vector
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%   - This function replaces diagonal entries of the co-occurrence matrix 
%     so that the result will exhibit low-rank structure.
%
function [C, values, elapsedTime] = rectifyC_DC(C, K, T)
    % Determine the default stopping criteria.
    if nargin < 3
        T = 30;
    end

    % Get the sizing information.
    N = size(C, 1);
    N_half = floor(0.5*N);
    
    % Print out the initial status.
    fprintf('Start rectifying C...\n');     
    startTime = tic;    
    
    % Start iteration.
    values = zeros(T, N);
    for t = 1:int32(T)
        % Generate one permuation vector and randomly permute C.
        perm = randperm(N);
        C_perm = C(perm, perm);
        
        % Step 1: Compute the diagonal entries of upper-left block (C11).
        d1 = computeDiag(C_perm(1:N_half, 1:N_half), C_perm(1:N_half, N_half+1:N), K);
    
        % Step 2: Compute the diagonal entries of lower-right block (C22).
        d2 = computeDiag(C_perm(N_half+1:N, N_half+1:N), C_perm(N_half+1:N, 1:N_half), K);
    
        % Step 3: Save the diagonal vector in the original order.
        d = [d1' d2'];
        values(t, :) = d(perm);    
        
        % Report only the current iteration as the values are used differently.
       if mod(t, 1) == 0
           fprintf('- %d-th iteration...\n', t);
       end   
    end           
    
    % Step 4: Replace the diagonal entries of C by the median of completed diagonals.
    C(1:N+1:numel(C)) = median(values);
    
    % Step 5: Normalize to be joint-stochastic.
    C = C / sum(sum(C));
    
    % Print out the final status.
    elapsedTime = toc(startTime);
    fprintf('+ Finish diagonal completion!\n');
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);          
end


%%
% Inner: computeDiag()
%
% Inputs: 
%   - F: (N/2)x(N/2) diagonal block matrix of co-occurrence matrix
%   - G: (N/2)x(N/2) off-diagonal block matrix of co-occurrence matrix
%   - K: the low-rank to enforce
%
% Outputs:
%   - d: (N/2)x1 vector of diagonal elements
%
% Remarks: 
%   - Let Fj be the j-th column vector of on-diagonal block matrix 
%   - Let I be the all indices except the j (i.e., [1:j-1, j+1:N/2])
%   - Then our goal is to find a K dimensional vector y such that U(I)y =
%     F_j(I), making F_j(I) dwell in the range space of U
%
%   - The normal equations for the least square problems of finding y is
%     (U(I)'U(I))y = U(I)'F_j(I), thus y = (U(I)'U(I))^(-1) U(I)'F_j(I)

%   - Saying I_K = KxK identity matrix and u_j is j-th row vector of U, 
%     U(I)'*U(I) = I_K - u_j'*u_j (By the property of orthogonal matrix U)

%   - Thus inv(U(I)'U(I)) = inv(I_K - u_j'*u_j). By Sherman-Morrison, it is
%     I_K + (u_j'*u_j)/(1 - u_j*u_j'), without requiring actual inverse
%
%   - Putting P_j := U(I)'F_j(I) (Kx1 column vector) and after expansion,
%     y = P_j + ((u_j'*u_j)/(1 - u_j*u_j'))P_j 
%
%   - Finally, d_j = (Uy)_j = (u_j)*y 
%                  = u_j*P_j + ((u_j*u_j'*u_j)/(1 - u_j*u_j'))P_j
%                  = u_j*P_j / (1 - u_j*u_j')
%
function d = computeDiag(F, G, K)
    % Step 1: Perform sigular-vector decomposition and truncate the columns.
    [U, ~, ~] = svd(G);
    U = U(:, 1:K);
    
    % Step 2: Precompute P where j-th column vector P_j = U(I)'*F_j(I).
    % Note that F - diag(diag(F)) is a matrix F without diagonal entries.
    P = U'*(F - diag(diag(F)));
    
    % Step 3: Compute u_j*P_j by entry-wise multiplication and column-sum.
    u_j_mul_P_j = sum(U.*P', 2);
    
    % Step 4: Compute u_j*u_j' by entry-wise multiplication and column-sum.
    u_j_mul_u_j = sum(U.*U, 2);
    
    % Step 5: Compute the vector of diagonal entries
    d = u_j_mul_P_j ./ (1 - u_j_mul_u_j);
end




%%
% TODO:
%
