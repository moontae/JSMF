%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%


%%
% Main: solveSCLS_expGrad()
%
% Inputs:
%   - G: KxK precomputed matrix, which is U'U (U: NxK matrix)
%   - f: Kx1 precomputed vector, which is U'v (v: Nx1 vector)
%   - y0: Kx1 initial solution vector (default = (1/K, ..., 1/K))
%   - T: the maximum iterations (default = 500)
%   - tolerance: the stopping criteria which is a duality gap between (default = 0.00001)
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
function [y, isConverged] = solveSCLS_expGrad(G, f, y0, T, tolerance, eta)
    % Set the default learning rate.
    if nargin < 6
        eta = 50.0;
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
    
    % Compute the gradient vector using the invariant parts.
    % J(y)      = ||Uy - v||^2 = (Uy - v)'(Uy - v) = (y'U' - v')(Uy - v)
    %           = y'U'Uy - 2(v'U)y + ||v||^2 (positive scalar)
    % grad_y(J) = 2U'Uy - 2(v'U)' = 2(U'Uy - U'v) (Kx1 column vector)
    gradients = 2.0*(G*y - f);
        
    % Perform update steps until convergence.
    for t = 1:T
        % Step 1a: Perform component-wise multiplicative update in the original space.
        y = y .* exp(-eta * gradients);
        
        % Step 1b: Project onto the K-dimensional simplex.
        y = y / norm(y, 1);
        
        % Step 2a: Evaluate the gradient.
        gradients = 2.0*(G*y - f);
                
        % Step 2b: Compute nu which makes mu dual-feasible. (i.e., mu >= 0)
        nu = -min(gradients);
        
        % Step 2c: Compute the feasible dual variable mu
        mu = gradients + nu;
        
        % Step 3: Check the duality gap (i.e., complementary slackness)
        dualityGap = mu'*y;
        if dualityGap < tolerance
            % If every component of duality gap is less than the tolerance,
            isConverged = 1;
            break
        end        
    end
end




%%
% TODO:
%

