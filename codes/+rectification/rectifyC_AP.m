%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [C_rect, values] = rectifyC_AP(C, 100);
%   - [C_rect, values] = rectifyC_AP(C, 100, 5);
%


%%
% Main: rectifyC_AP()
%
% Inputs:
%   - C: NxN co-occurrence matrix (joint-stochastic)
%   - K: the number of basis vectors (the number of topics)
%   - T: the number of maximum iterations (default = 15)
%
% Outputs:
%   - C: NxN co-occurrence matrix (joint-stochastic & doubly-nonnegative)
%   + values: 2xT statistics
%     - 1st row: Changes between before and after iteration in terms of Frobenius norm
%     - 2nd row: Average square difference betweeo before and after projections in terms of Frobenius norm 
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%   - This function performs a series of projections alternating between
%     two convex sets: non-negative & joint-stochastic matrices and
%     one non-convex set: positive semidefinite matrices with rank K.
%
function [C, values, elapsedTime] = rectifyC_AP(C, K, T)
    % Set the default number of iterations.
    if nargin < 3
        T = 150;        
    end
    
    % Prepare for the projection.
    values = zeros(2, T);
    C_NN = C;
    
    % Print out the initial status.
    fprintf('[rectification.rectifyC_AP] Start rectifying C...\n'); 
        
    % Perform alternating projection.
    startTime = tic;
    for t = 1:int32(T)
       % Backup the previous C.
       C_prev = C_NN;
                           
       % Perform one iteration of alternating projection.
       C_PSD = nearestPSD(C_NN, K);
       d_PSD = norm(C - C_PSD, 'fro');
       
       C_JS  = nearestJS(C_PSD);
       d_JS  = norm(C - C_JS, 'fro');
       
       C_NN  = nearestNN(C_JS);
       d_NN  = norm(C - C_NN, 'fro');
              
       % Compute the convergence statistics.
       values(1, t) = norm(C_NN - C_prev, 'fro');
       values(2, t) = (d_PSD^2 + d_JS^2 + d_NN^2) / 6;
       if mod(t, 1) == 0
           fprintf('- iteration %d... (%e / %e)\n', t, values(1, t), values(2, t));
       end              
    end    
    
    % Perform a post-hoc normalization to make the matrix joint-stochastic.
    % (This step is not necessary but for uniform experiments)
    C = C_NN / sum(sum(C_NN));
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish alternating projection!\n');    
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);           
end


%%
% Inner: nearestNN()
%
% Inputs: 
%   - C: NxN co-occurrence matrix
%
% Outputs:
%   - C: NxN non-negative matrix
%
% Remarks:
%
function C = nearestNN(C)
    C = max(C, 0);
end


%%
% Inner: nearestJS()
%
% Inputs: 
%   - C: NxN co-occurrence matrix
%
% Outputs:
%   - C: NxN joint-stochastic matrix
%
% Remarks:
%
function C = nearestJS(C)
    N = size(C, 1);
    C = C + (1 - sum(sum(C)))/(N^2);
end


%%
% Inner: nearestPSD()
%
% Inputs: 
%   - C: NxN co-occurrence matrix
%   - K: the number of non-negative eigenvalues to use
%
% Outputs:
%   - C: NxN positive semidefinite matrix
%
% Remarks:
%   - This function projects the given matrix into the convex set of
%     positive semidefinite matrices with the rank K
%   - Due to epsilon numerical error, it symmetrize the matrices
%
function C = nearestPSD(C, K)
    % Find the nearest positive semidefinite matrix with the rank K.
    [V, D] = eigs(C, K, 'LA');
    C = V * diag(max(diag(D), 0)) * V';
    C = 0.5*(C + C');
end




%%
% TODO:
%
