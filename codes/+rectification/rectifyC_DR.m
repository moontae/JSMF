%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Sungjun Cho & Moontae Lee
% Examples:
%   - [C_rect, values] = rectifyC_DR(C, 100);
%   - [C_rect, values] = rectifyC_DR(C, 100, 5);
%


%%
% Main: rectifyC_DR()
%
% Inputs:
%   - C: NxN co-occurrence matrix (joint-stochastic)
%   - K: the number of basis vectors (the number of topics)
%   - T: the number of maximum iterations (default = 15)
%
% Outputs:
%   - C: NxN co-occurrence matrix (joint-stochastic & doubly-nonnegative)
%   + values: 2xT statistics
%     - 1st row: changes between before and after iteration in terms of Frobenius norm
%     - 2nd row: average square difference between before and after projections in terms of Frobenius norm 
%   - elapsedTime: total elapsed amount of seconds
%
% Remarks: 
%   - This function performs cyclic Douglas-Rachford iterations onto
%     two convex sets: non-negative & joint-stochastic matrices and
%     one non-convex set: positive semidefinite matrices with rank K.
%   - Each function implemented below represents a 2-set Douglas-Rachford 
%     operator that is repeatedly used within the DR iteration scheme.
%   - Given two sets A and B, the operator is defined as (I + R_B R_A)/2
%     where R_A denotes the reflection with respect to A.
%
function [C, values, elapsedTime] = rectifyC_DR(C, K, T)
    % Set the default number of iterations.
    if nargin < 3
        T = 150;        
    end
    
    % Prepare for the projection.
    values = zeros(2, T);
    C_3 = C;
    
    % Print out the initial status.
    fprintf('[rectification.rectifyC_DR] Start rectifying C...\n'); 
        
    % Perform Douglas-Rachford projection.
    startTime = tic;
    for t = 1:int32(T)
       % Backup the previous C.
       C_prev = C_3;
                           
       % Perform one iteration of Douglas-Rachford projection.
       C_1 = projectPSD_JS(C_3, K);
       d_1 = norm(C - C_1, 'fro');
       
       C_2  = projectJS_NN(C_1);
       d_2  = norm(C - C_2, 'fro');
       
       C_3  = projectNN_PSD(C_2, K);
       d_3  = norm(C - C_3, 'fro');
              
       % Compute the convergence statistics.
       values(1, t) = norm(C_3 - C_prev, 'fro');
       values(2, t) = (d_1^2 + d_2^2 + d_3^2) / 6;
       if mod(t, 1) == 0
           fprintf('- iteration %d... (%e / %e)\n', t, values(1, t), values(2, t));
       end              
    end    
    
    % Perform a post-hoc normalization to make the matrix joint-stochastic.
    % (This step is not necessary but for uniform experiments)
    C = C_3 / sum(sum(C_3));
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish Douglas-Rachford projection!\n');    
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);           
end


%%
% Inner: projectPSD_JS()
%
% Inputs: 
%   - C: NxN co-occurrence matrix
%   - K: the number of non-negative eigenvalues to use
%
% Outputs:
%   - C: NxN co-occurrence matrix
%
function C = projectPSD_JS(C, K)
    R_PSD = 2*nearestPSD(C, K) - C;
    R_JS = 2*nearestJS(R_PSD) - R_PSD;
    C = (C + R_JS)/2; 
end


%%
% Inner: projectJS_NN()
%
% Inputs: 
%   - C: NxN co-occurrence matrix
%
% Outputs:
%   - C: NxN co-occurrence matrix
%
function C = projectJS_NN(C)
    R_JS = 2*nearestJS(C) - C;
    R_NN = 2*nearestNN(R_JS) - R_JS;
    C = (C + R_NN)/2;  
end


%%
% Inner: projectNN_PSD()
%
% Inputs: 
%   - C: NxN co-occurrence matrix
%   - K: the number of non-negative eigenvalues to use
%
% Outputs:
%   - C: NxN co-occurrence matrix
%
function C = projectNN_PSD(C, K)
    R_NN = 2*nearestNN(C) - C;
    R_PSD = 2*nearestPSD(R_NN, K) - R_NN;
    C = (C + R_PSD)/2;
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


