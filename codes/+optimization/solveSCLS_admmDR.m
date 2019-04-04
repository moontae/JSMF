%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%


%%
% Main: solveSCLS_admmDR()
%
% Inputs:
%   - UtU: KxK precomputed matrix, which is U'U (U: NxK matrix)
%   - Utv: Kx1 precomputed vector, which is L'l (v: Nx1 vector)
%   - T: the maximum iterations (default = 500)
%   - tolerance: the stopping criteria for dulatiy gap (default = 0.00001)
%
% Outputs:
%   - y: Kx1 column non-negtive vector whose 1-norm is 1
%   - isConverged: 1 if the algorithm converges, 0 otherwise
%
% Remarks:
%   - This function finds a least square vector y that minimizes ||Uy - v||^2 + Lambda*(yy^T - V) 
%     where the given KxK matrices: Lambda (multipliers) and V (regularizer).
%   - The solution y must be a non-negative vector on the K-dim simplex.
%   - It receives the precomputed invarainats rather than the raw (U, v).
%
function [y, isConverged] = solveSCLS_admmDR(F, f, lambda, y0, T, tolerance)
    % Set the default tolerance.
    if nargin < 6
        tolerance = 0.00001;
    end
    
    % Set the default number of iterations.
    if nargin < 5
        T = 500;
    end
       
    % Initailize variables.
    if nargin < 4
        K = size(F, 1);
        y = (1/K)*ones(K, 1);
    else
        y = y0;
    end
    isConverged = 0;
    b = y;
    
    % Perform update steps until convergence.    
    prox = @(x) F*(x + f);
    for t = 1:T
        prev_y = y;
        a = prox(2*y - b);
        b = b + lambda*(a - y);
        y = optimization.projectToSimplex(b);
        
        if norm(y - prev_y, 2) < tolerance            
            isConverged = 1;
            break
        end        
    end    
end
  



%%
% TODO:
%
