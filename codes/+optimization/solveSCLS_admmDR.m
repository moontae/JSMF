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
%   - G: KxK precomputed matrix, which is inv(gamma*U'U + I_k) (U: NxK matrix)
%   - f: Kx1 precomputed vector, which is gamma*U'U (v: Nx1 vector)
%   - y0: Kx1 initial solution vector (default = (1/K, ..., 1/K))
%   - T: the maximum iterations (default = 500)
%   - tolerance: the stopping criteria which is 2-norm change in consecutive solutions (default = 0.00001)
%
% Outputs:
%   - y: Kx1 column non-negtive vector whose 1-norm is equal to 1
%   - isConverged: 1 if the algorithm converges, 0 otherwise
%
% Remarks:
%   - This function finds a least-square solution y that minimizes ||Uy - v||^2 
%     with the simplex constraint.
%   - Users should feed the precomputed invarainats rather than inputting U and v.
%   - One can easily extend the objective function to ||Uy - v||^2 + <Lambda, yy^T - V>_F
%     just by feeding an augmented F.
%
function [y, isConverged] = solveSCLS_admmDR(G, f, y0, T, tolerance, lambda)
    % Set the default parameter.
    if nargin < 6
        lambda = 1.9;
    end

    % Set the default tolerance.
    if nargin < 5
        tolerance = 0.00001;
    end
    
    % Set the default number of iterations.
    if nargin < 4
        T = 500;
    end
       
    % Set an initial solution and other variables.
    if nargin < 3
        K = size(G, 1);
        y = (1/K)*ones(K, 1);
    else
        y = y0;
    end
    isConverged = 0;
    q = y;
    
    % Perform update steps until convergence.    
    prox = @(x) G*(x + f);
    for t = 1:T
        % Store the previous solution.
        prev_y = y;
        
        % Update the existing solution.
        p = prox(2*y - q);
        q = q + lambda*(p - y);
        y = optimization.projectToSimplex(q);
        
        % Check the stopping criteria.
        if norm(y - prev_y, 2) < tolerance            
            isConverged = 1;
            break
        end        
    end    
end
  



%%
% TODO:
%
