%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%


%%
% Main: rectify_PALM()
%
% Remark: 
%   - Compress + Rectify by Proximal Alternating Linearized Minimization
%
function [Y1, Y2, values, elapsedTime] = rectify_PALM(C, K, r, T, V, D)
    % Set the default number of iterations.
    if nargin < 4
        T = 100;
    end      

    % Set the default r.
    if nargin < 3
        r = 1e-4;
    end
        
    % Print out the initial status.
    fprintf('Start compressing + rectifying by PALM...\n');
    
    % Initialize by truncated eignedecomposition if not specified in the arguments.
    if nargin < 6
        [V, D] = eigs(C, K, 'LA');
    end    
    Y1 = V*sqrt(D);
    Y2 = Y1;
        
    % Set the parameter value.
    gamma = 2.01;
    I = eye(K, K);    
    
    % For each PALM iteration,
    startTime = tic;
    values = zeros(2, T);    
    for t = 1:int32(T)
        % Update Y1 part.       
        c = gamma*norm(Y2'*Y2 + r*I, 2);
        U = Y1 - 2*((Y1*Y2' - C)*Y2 + r*(Y1-Y2))/c;       
        value1 = norm(max(U, 0) - Y1, 'fro');
        Y1 = max(U, 0);
       
        % Update Y2 part.
        d = gamma*(norm(Y1'*Y1 + r*I, 2));
        V = Y2 - 2*((Y2*Y1' - C)*Y1 + r*(Y2-Y1))/d;
        value2 = norm(max(V, 0) - Y2, 'fro');
        Y2 = max(V, 0);
       
        % Compute the convergence statistics.
        values(1, t) = 0.5*(value1 + value2);
        values(2, t) = 0.5*norm(C-Y1*Y2','fro') + 0.5*r*norm(Y1-Y2,'fro');
        if (mod(t, 1) == 0)
            fprintf('- iteration %d... (%e, %e)\n', t, values(1, t), values(2, t));
        end         
    end    
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish PALM!\n');    
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);           
end




%%
% TODO:
%






