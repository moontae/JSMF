%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Kun Dong & Moontae Lee
% Examples:
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR] = factorizeY(C, 100);
%   - [S, B, A, Btilde, Cbar, C_rowSums, diagR] = factorizeY(C, 100, 0);
%   


%%
% Main: factorizeY()
%
% Inputs:
%   - Y: NxK rectified and compressed co-occurrence  (could consist of negative entries)
%   - K: the number of basis vectors (i.e., the number of topics)
%   + optimizer: the option for learning topics
%     - 'expGrad':   learn bases (i.e., topics) by the Exponentiated-Gradient algorithm
%     - 'admmDR':    learn bases (i.e., topics) by the ADMM with Douglas-Rachford splitting
%     - 'activeSet': learn bases (i.e., topics) by the Active Set algorithm
%
% Outputs:
%   - S:           1xK column vector having the basis indices
%   - B:           NxK object-cluster matrix where B_{nk} = p(X=n | Z=k) 
%   - A:           KxK cluster-cluster matrix where A_{kl} = p(Z1=k, Z2=l)
%   - Btilde:      KxN cluster-object matrix where Btilde_{kn} = p(Z=k | X=n) 
%   - Cbar:        NxN approximated row-normalized co-occurrence matrix where Cbar_{ij} = p(X2=j | X1=i)
%   - C_rowSums:   Nx1 approximated row-sum of co-occurrence matrix where C_rowSums_i = p(X=i)
%   - diagR:       1xK vector indicating the scores of each basis vector
%   - C:           NxN updated C matrix after the rectification step
%   - values:     (Different values from the factorizeC's. Not compatible)
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%   - This function performs the Rectified Anchor Word algorithm given the
%     rectified and compressed co-occurrence matrix Y and the number of bases.
%  
function [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values, elapsedTime] = factorizeY(Y, K, optimizer, dataset)    
    % Set the default parameters.
    if nargin < 4
        dataset = '';
    else
        dataset = sprintf('_%s', dataset);
    end        
    if nargin < 3        
        optimizer = 'activeSet';
    end
    
    % Prepare a logger to record the result and performance.    
    logger = logging.getLogger('factorizeY_logger', 'path', sprintf('factorizeY%s_K-%d_%s.log', dataset, K, optimizer));
    logger.info('factorizeY');
    beginTime = tic;
    
    
    %------------------------------------------------------------------------------------------------
    % Step 0: Prepares variables based on the tall compression Y that approximately satisfies C=YY^T.
    
    % Compute the approximated column-sum vector of C based only on Y.
    % d = (e^T)C ~= (e^T)YY^T = (e^T*Y)Y^T = (column-sum of Y)*Y^T
    d = sum(Y)*Y';
    
    % Define Ybar = diag(d)^(-1) * Y, then Ybar^T = Y^T*diag(d)^(-1).
    Ybart = bsxfun(@rdivide, Y', d);
        

    %------------------------------------------------------------------------------------------------
    % Step 1. Find S based only on Y.
    logger.info('+ Start finding the set of anchor bases S...');    
    startTime = tic;
    
    % Run a plain QR-decomposition without the column-pivoting.
    [Q, R] = qr(Y, 0);
    
    % Column-normalized C satisfies Cbart = YY^T*diag(d)^(-1) = Y*Ybar^T = QR*Ybar^T = Q*(R*Ybar^T).
    RYbart = R*Ybart;      
    
    % Note that QR with column-pivoting on Cbart is equivalent to QR with column-pivoting on R*Ybar^T.
    % This is because Q is an orthogonal transformation such as a reflection, a rotation, or a permutation.
    [~, R1, S] = qr(RYbart, 'vector');
    diagR = abs(diag(R1))';
    
    % Extract values corresponding to only the given number of basis vectors.
    S = S(1:K);
    diagR = diagR(1:K);    
    
    % Print the output information.
    elapsedTime = toc(startTime);
    logger.info('  - Finish finding S! [%f]', elapsedTime);    
    
    
    %------------------------------------------------------------------------------------------------
    % Step 1.5: Add non-negative corrections if anchor rows consist of negative entries.
    logger.info('+ Start evaluating the non-negative corrections E...');    
    startTime = tic;
    
    % Retrieve the KxN submatrix corresponding to the anchor rows from the approximated C.
    C_S = Y(S, :)*Y';
    
    % Find the (rows, cols, values) of the non-negative entries.
    % Prepare a sparse correction matrix E.
    N = size(Y, 1);
    [rows, cols] = find(C_S < 0);
    values = C_S(C_S < 0);
    E = sparse(S(rows), cols, -values, N, N);
    E = max(E, E');
    
    % Adjust the column-sum vector of C by the correction.
    % Recompute the Ybart with the corrected d.
    d = d + sum(E);        
    Ybart = bsxfun(@rdivide, Y', d);
    
    % Prepare a correction for normalized co-occurrence matrix.
    Ebar = E ./ d;
    RYbart = R*Ybart + Q'*Ebar;
    
    % Print the output information.
    elapsedTime = toc(startTime);
    logger.info('  - Finish evaluating E! [%f]', elapsedTime);
        
    
    %------------------------------------------------------------------------------------------------
    % Step 2. Recover B based on Y and S.   
    logger.info('+ Start recovering the object-cluster B...');    
    startTime = tic;
    
    % Initailize and prepares return variables.
    N = size(Y, 1);
    Btilde = zeros(K, N);
    convergences = zeros(1, N);
    
    % Precompute the invariant parts.    
    U = RYbart(:, S);
    Ut = U';
    UtU = Ut*U;      
        
    % Perform the trivial inference for the basis vectors.
    Btilde(:, S) = eye(K);
    convergences(S) = 1;
            
    % Perform the main inference for the non-basis vectors.    
    switch(optimizer)
      case 'expGrad'
        % For each row (in parallel),
        parfor n = 1:int32(N)
            % Skips the basis vectors.
            if any(n == S)
                continue
            end
            
            % If the given member is not a basis basis vector,
            v = RYbart(:, n);
            Utv = Ut*v;                   
            [y, isConverged] = optimization.solveSCLS_expGrad(UtU, Utv);            
            
            % Save the recovered distribution p(z | x=n) and convergence.
            Btilde(:, n) = y;
            convergences(n) = isConverged;

            % Print out the progress for each set of objects.
            if mod(n, 500) == 0
                fprintf('  - %d-th member...\n', n);
            end
        end
        
      case 'admmDR'
        gamma = 3.0;  
        lambda = 1.9;        
        
        % Precompute the invariant parts.
        F = inv(gamma*UtU + eye(K, K));        
                
        % For each row (in parallel), 
        parfor n = 1:int32(N)
            % Skip the basis vectors.
            if any(n == S)
                continue
            end
            
            % If the given member is not a basis basis vector,
            v = RYbart(:, n);
            Utv = Ut*v;                        
            f = gamma*Utv;        
            y0 = optimization.projectToSimplex(UtU\Utv);
            [y, isConverged] = optimization.solveSCLS_admmDR(F, f, lambda, y0);            
            
            % Save the recovered distribution p(z | x=n) and convergence.
            Btilde(:, n) = y;
            convergences(n) = isConverged;

            % Print out the progress for each set of objects.
            if mod(n, 500) == 0
                fprintf('  - %d-th member...\n', n);
            end
        end
        
      case 'activeSet'
        % For each row (in parallel), 
        parfor n = 1:int32(N)
            % Skip the basis vectors.
            if any(n == S)
                continue
            end
              
            % If the given member is not a basis basis vector,
            v = RYbart(:, n);
            y = optimization.solveSCLS_activeSet(U, v);
              
            % Save the recovered distribution p(z | x=n) and convergence.
            Btilde(:, n) = y;
              
            % Print out the progress for each set of objects.
            if mod(n, 500) == 0
                fprintf('  - %d-th member...\n', n);
            end
        end
    end

    % Fills out the output variables.
    % Recall that Cbart = YY^T*diag(d)^(-1) = Y*Ybar^T = QR*Ybar^T = Q*(R*Ybar^T).
    C_rowSums = d';    
    Cbar = (Y*Ybart + Ebar)';
    C = sparse(Y*Y' + E);   
    
    % Recover topics B based on the learned topic-word matrix Btilde.
    denominators = 1.0 ./ (Btilde * C_rowSums);
    B = Btilde' .* (C_rowSums * denominators');
    loss = norm(Q*U*Btilde - Cbar, 'fro');   
    
    % Print the output information.
    elapsedTime = toc(startTime);         
    logger.info('  + Finish recovering B! [%f]', elapsedTime);
    logger.info('    - %d/%d objects are converged by [%s].', sum(convergences), N, optimizer);
    logger.info('    - loss = %.4f (By Frobenius norm)', loss);
        
    
    %------------------------------------------------------------------------------------------------
    % Step 3. Recover A based on Y, S, and B.   
    logger.info('+ Start recovering the cluster-cluster A...');
    startTime = tic;
    
    % Since C = YY', C(S, S) = Y(S, :)*Y'(:, S)
    C_SS = Y(S, :)*Y(S, :)' + E(S, S);
    B_S = B(S, :);
    invB_S = diag(1.0 ./ diag(B_S));
    
    % Perform a diagonal recovery first.
    % Note that the result of the diagonal recovery provides initializations for optimizations.
    A = invB_S*C_SS*invB_S;    
    
    % Print the output information.
    elapsedTime = toc(startTime);    
    logger.info('  - Finish recovering A! [%f]', elapsedTime);
    
    
    % Print out the overall infromation.
    elapsedTime = toc(beginTime);    
    logger.info('- Finish factorizing Y! [%f]', elapsedTime);    
    
    % Close the logger.
    logging.clearLogger('factorizeY_logger');
end




%%
% TODO:
%


