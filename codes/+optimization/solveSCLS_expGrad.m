%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%


%%
% Main: solveSCLS_expGrad()
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
%   - This function finds a least square vector y that minimizes ||Uy - v||^2.
%   - The solution y must be a non-negative vector on the K-dim simplex.
%   - It receives the precomputed invarainats rather than the raw (U, v).
%
function [y, isConverged] = solveSCLS_expGrad(UtU, Utv, T, tolerance)
    % Set the default tolerance.
    if nargin < 4
        tolerance = 0.00001;
    end
    
    % Set the default number of iterations.
    if nargin < 3
        T = 500;
    end
       
    % Initailize variables.
    K = size(UtU, 1);
    y = (1/K)*ones(K, 1);
    isConverged = 0;
    
    % Compute the gradient vector using the invariant parts.
    % J(y)      = ||Uy - v||^2 = (Uy - v)'(Uy - v) = (y'U' - v')(Uy - v)
    %           = y'U'Uy - 2(v'U)y + ||v||^2 (positive scalar)
    % grad_y(J) = 2U'Uy - 2(v'U)' = 2(U'Uy - U'v) (Kx1 column vector)
    gradients = 2.0*(UtU*y - Utv);
        
    % Perform update steps until convergence.
    eta = 50.0;
    for t = 1:T
        % Step 1a: Perform component-wise multiplicative update in the original space.
        y = y .* exp(-eta * gradients);
        
        % Step 1b: Project onto the K-dimensional simplex.
        y = y / norm(y, 1);
        
        % Step 2a: Evaluate the gradient.
        gradients = 2.0*(UtU*y - Utv);
                
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

