%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%


%%
% Main: rectify_IPALM()
%
% Inputs:
%   - C: NxN original (joint-stochastic) co-occurrence matrix 
%   - K: the number of basis vectors
%   - r: the relative weight of X=Y constraint comparing to C=XY
%   - T: the number of iteration
%   - V: user-specified initialization similar to eigenvectors
%   - D: user-specified intiialization similar to eigenvalues
%
% Outputs:
%   - Y1: NxK non-negative matrix (often called by X)
%   - Y2: NxK non-negative matrix (often called by Y) that approximates C by XY'
%
% Remark: 
%   - Compress x Rectify by Intertial Proximal Alternating Linearized Minimization
%   + This function tries to solve the following minimization problem:
%       - minimize (||C - XY'||_F)^2 + r(||X - Y||_F)^2
%       - subject to X >= 0 and Y >= 0
%   - Note that another objective/coinstraint e'XYe = 1 is subsummed by other two parts.
%
function [Y1, Y2, values, elapsedTime] = rectify_IPALM(C, K, r, T, V, D)    
    % Set the default number of iterations.    
    if nargin < 4
        T = 100;
    end    

    % Set the default r.
    if nargin < 3
        r = 1e-4;
    end
        
    % Print out the initial status.
    fprintf('[compression.rectify_IPALM] Start compressing + rectifying by IPALM...\n'); 
    fprintf('- Relative weight r = %f\n', r);    

    % Initialize by truncated eignedecomposition if not specified in the arguments.
    if nargin < 6
        [V, D] = eigs(C, K, 'LA');
    end    
    Y1 = V*sqrt(D);
    Y1_prev = Y1;
    Y2 = Y1;
    Y2_prev = Y2;
    
    % Prepare basic constants.
    I = eye(K, K);    
    
    % For each iPALM iteration,
    startTime = tic;
    values = zeros(2, T);
    for t = 1:T
        % Update the Y1 part.
        alpha1 = 0.1;
        beta1 = 0.1;
        c = norm(Y2'*Y2 + r*I, 2);
        tau1 = ((1 + 2*beta1) / (2 - 2*alpha1)) * c;
        U1 = Y1 + alpha1*(Y1 - Y1_prev);
        V1 = Y1 + beta1*(Y1 - Y1_prev);
        Y1_prev = Y1;
        Y1 = max(U1 - ((V1*Y2' - C)*Y2 + r*(V1-Y2))/tau1, 0);
        value1 = norm(Y1 - Y1_prev, 'fro');
      
        % Update the Y2 part.
        alpha2 = 0.1;
        beta2 = 0.1;
        d = norm(Y1'*Y1 + r*I, 2);
        tau2 = ((1 + 2*beta2) / (2 - 2*alpha2)) * d;
        U2 = Y2 + alpha2*(Y2 - Y2_prev);
        V2 = Y2 + beta2*(Y2 - Y2_prev);
        Y2_prev = Y2;
        Y3 = max(U2 - ((V2*Y1' - C)*Y1 + r*(V2-Y1))/tau2, 0);       
        value2 = norm(Y2 - Y2_prev, 'fro');
       
        % Compute the convergence statistics.
        values(1, t) = 0.5*(value1 + value2);
        values(2, t) = norm(C - Y1*Y2', 'fro');
        if (mod(t, 1) == 0)
            fprintf('- iteration %d... (%e, %e) / (%e, %e)\n', t, c, d, values(1, t), values(2, t));
        end   
    end        
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish IPALM!\n');    
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);         
end




%%
% TODO:
%

