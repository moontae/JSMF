%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [B, Btilde] = recoverB(Cbar, C_rowSums, S);
%


%%
% Main: recoverB()
%
% Inputs:
%   - Cbar: NxN row-normalized co-occurrence matrix
%   - C_rowSums: Nx1 vector having sums of each row in original C matrix
%   - S: 1xK vector having the row indices of approximate basis vectors
%   + option:
%     - expGrad: 
%     - admmDR: 
%     - activeSet: 
%
% Intermediates:
%   - H: NxK matrix having where each column is a basis vector (H = Cbar_S')
%   - h: Nx1 column vector indicating a non-basis vector 
%   - y: Kx1 column vector, non-negative least square solution in the simplex
%
% Outputs:
%   - B: NxK object-cluster tall matrix where B_{nk} = p(X=n | Z=k) 
%   - Btilde: KxN cluster-object fat matrix where Btilde_{kn} = p(Z=k | X=n) 
%   - elapsedTime: total elapsed amount of seconds
%
function [B, Btilde, elapsedTime] = recoverB(Cbar, C_rowSums, S, option)
    % Set the default option.
    if nargin < 4
        option = 'activeSet';
    end    

    % Initailize dimensional and return variables.
    N = size(Cbar, 1);
    K = numel(S);        
    Btilde = zeros(K, N);
    convergences = zeros(1, N);
    
    % Precompute the invariant parts.
    U = Cbar(S, :)';        
    UtU = U'*U;    
    
    % Print out the initial status.
    fprintf('[inference.recoverB] Start recovering the object-cluster B...\n');
    
    % Compute the Btilde (for each member object in parallel).    
    startTime = tic;
    
    % Perform the trivial inference for the basis vectors.
    Btilde(:, S) = eye(K);
    convergences(S) = 1;
        
    % Perform the main inference for the non-basis vectors.    
    switch (option)
      case 'expGrad'
        % For each row (replace for to parfor for parallel running),        
        for n = 1:int32(N)
            % Skip the basis vectors.
            if any(n == S)
                continue
            end
            
            % If the given member is not a basis basis vector,   
            v = Cbar(n, :)';
            Utv = U'*v;                   
            [y, isConverged] = optimization.solveSCLS_expGrad(UtU, Utv);            
            
            % Save the recovered distribution p(z | x=n) and convergence.
            Btilde(:, n) = y;
            convergences(n) = isConverged;

            % Print out the progress for each set of objects.
            if mod(n, 500) == 0
                fprintf('  - %d-th object...\n', n);
            end
        end
        
      case 'admmDR'        
        gamma = 3.0;
        
        % Precompute the invariant parts.
        G = inv(gamma*UtU + eye(K, K));        
                
        % For each row (replace for to parfor for parallel running),
        for n = 1:int32(N)
            % Skip the basis vectors.
            if any(n == S)
                continue
            end
            
            % If the given member is not a basis basis vector,    
            v = Cbar(n, :)';
            Utv = U'*v;                        
            [y, isConverged] = optimization.solveSCLS_admmDR(G, gamma*Utv, optimization.projectToSimplex(UtU\Utv));
                        
            % Save the recovered distribution p(z | x=n) and convergence.
            Btilde(:, n) = y;
            convergences(n) = isConverged;

            % Print out the progress for each set of objects.
            if mod(n, 500) == 0
                fprintf('  - %d-th object...\n', n);
            end
        end        
           
      case 'activeSet'
        % For each row (replace for to parfor for parallel running),
        for n = 1:int32(N)
            % Skip the basis vectors.
            if any(n == S)
                continue
            end
            
            % If the given member is not a basis basis vector,
            v = Cbar(n, :)';
            y = optimization.solveSCLS_activeSet(U, v);
            
            % Save the recovered distribution p(z | x=n) and convergence.
            Btilde(:, n) = y;
            
            % Print out the progress for each set of objects.
            if mod(n, 500) == 0
                fprintf('  - %d-th object...\n', n);
            end
        end    
    end

    % Recover the B after finishing simplex NNLS in parallel.
    denominators = 1.0 ./ (Btilde * C_rowSums);
    B = Btilde' .* (C_rowSums * denominators');
    elapsedTime = toc(startTime);
    
    % Print out the final status
    loss = norm(U*Btilde - Cbar, 'fro');
    fprintf('+ Finish recovering B!\n');
    fprintf('  - %d/%d objects are converged by [%s].\n', sum(convergences), N, option);
    fprintf('  - Loss = %.4f (By Frobenius norm)\n', loss);
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);      
end




%%
% TODO:
%

