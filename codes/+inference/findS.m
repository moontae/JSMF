%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [S, diagR] = findS(Cbar, 100);
%   - [S, diagR] = findS(Cbar, 100, 'densePartial');
%   - [S, diagR] = findS(Cbar, 100, 'sparsePartial');
%   - [S, diagR] = findS(Cbar, 100, 'full');
%   - [S, diagR] = findS(Cbar, 3,   'pca');
%   - [S, diagR] = findS(Cbar, 3,   'tsne');
%   - [S, diagR] = findS(Cbar, 100, 'influenceScore');
%


%%
% Main: findS()
%
% Inputs:
%   - Cbar: NxN row-normalized co-occurrence matrix (row-stochastic)
%   - K: the number of basis vectors / the low-dimension for pca/tsne
%   + option: method to find basis vectors (default = 'sparsePartial')
%     - densePartial: 
%     - sparsePartial:
%     - full:
%     - pca:
%     - tSNE:
%     - influenceScore:
%
% Outputs:
%   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
%   - diagR: 1xK vector indicating the scores of each basis vector
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%   - This function performs QR-factorization with the row-pivoting, 
%     finding the given number of approximate nonnegative basis vectors.
%   - The 'full' method is useful only when N is small.
%   - The 'pca' finds the vertices of exact convex hull after linear projection.
%   - The 'tsne' finds the vertices of exact convex hull after non-linear projection.
%  
function [S, diagR, elapsedTime] = findS(Cbar, K, option)    
    % Set the default option.
    if nargin < 3
        option = 'sparsePartial';
    end
      
    % Print out the initial status.
    fprintf('[inference.findS] Start finding the set of anchor bases S...\n');     
        
    % Start finding the given number of approximate basis vectors.
    startTime = tic;
    switch (option)
      case 'densePartial'  
        [S, diagR] = densePartialQR(Cbar', K); 
      case 'sparsePartial' 
        [S, diagR] = sparsePartialQR(Cbar', K);
      case 'full'          
        [S, diagR] = fullQR(Cbar', K);
      case 'pca'  
        [S, diagR] = compressPCA(Cbar, K);
        K = numel(S);
      case 'tSNE' 
        [S, diagR] = compressTSNE(Cbar, K);
        K = numel(S);
      case 'influenceScore'
        [S, diagR] = influenceScore(Cbar', K);        
      otherwise
        error('  * Underfined option [%s] is given!\n', option);                  
    end    
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish finding S!\n');
    fprintf('  - Discovered %d basis anchor vectors by [%s] method.\n', K, option);
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);     
end


%%
% Inner: densePartialQR()
%
% Inputs:
%   - P: NxN column-normalized co-occurrence matrix (column-stochastic)
%   - K: the number of basis vectors 
%
% Outputs:
%   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
%   - diagR: 1xK vector indicating the distance to the current subspace
%
% Remarks: 
%   - This function greedily selects K column vectors of P by using
%     Gram-Schmidt process, which is equivalent to column-pivoting.
%   - It does not fully factorize P into QR, but project every non-basis
%     rows of P, loosing the sparse structure of P.
%
function [S, diagR] = densePartialQR(P, K)
    % Compute the squared sums for each column, and prepares return variables.
    colSquaredSums = sum(P.*P, 1);
    S = zeros(1, K);
    diagR = zeros(1, K);
    isColBasis = false(1, size(P, 2));
        
    % Start finding the given number of approximate basis vectors.
    for k = 1:int32(K)
        % Find the farthest column vector from the origin.
        [maxSquaredSum, maxCol] = max(colSquaredSums);
        S(k) = maxCol;
        diagR(k) = sqrt(maxSquaredSum);
        isColBasis(maxCol) = true;
    
        % Normalize the column vector corresponding to the current basis.
        P(:, maxCol) = P(:, maxCol) / diagR(k);
                
        % Project all other columns down to the orthogonal complement of 
        % the subspace spanned by the current set of basis vectors.
        innerProducts = P(:, maxCol)'*P;
        P = P - P(:, maxCol)*innerProducts;
        
        % Recompute the squared sums for every column (without exclusion).        
        colSquaredSums = sum(P.*P, 1);        
        
        % Ensure that the selected basis vectors are never chosen again.
        % (theoretically not necessary, but numerically safer)
        colSquaredSums(isColBasis) = 0;
    end    
end


%%
% Inner: sparsePartialQR()
%
% Inputs:
%   - P: NxN column-normalized co-occurrence matrix (column-stochastic)
%   - K: the number of basis vectors 
%
% Outputs:
%   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
%   - diagR: 1xK vector indicating the distance to the current subspace
%
% Remarks: 
%   - This function greedily selects K column vectors of P by using
%     Gram-Schmidt process, which is equivalent to column-pivoting.
%   - It does not fully factorize P into QR implicitly updating Q instead of
%     changing P (without losing the sparse structure of P).
%
function [S, diagR] = sparsePartialQR(P, K)
    % Compute the squared sums for each column and prepare return variables.
    colSquaredSums = sum(P.*P, 1);
    S = zeros(1, K);
    diagR = zeros(1, K);        
    isColBasis = false(1, size(P, 2));
        
    % Start finding the given number of approximate basis vectors.
    Q = zeros(size(P, 1), K);
    for k = 1:int32(K)
        % Find the farthest column vector from the origin.
        [maxSquaredSum, maxCol] = max(colSquaredSums);
        S(k) = maxCol;
        diagR(k) = sqrt(maxSquaredSum);
        isColBasis(maxCol) = true;
        
        % Compute the next basis q_n.
        Q(:, k) = P(:, maxCol);
        if (k > 1)
            % sumProjections = sum_{j=1}^{k-1} proj_{e_j}(p_max)
            sumProjections = Q(:, 1:k-1)*Q(:, 1:k-1)'*P(:, maxCol);
            
            % q_k = p_max - sum of projections to each basis
            Q(:, k) = Q(:, k) - sumProjections;
        end
        
        % Normalize the column vector corresponding to the current basis.
        Q(:, k) = Q(:, k) / diagR(k);
        
        % Update the suqared sums of column vector implicitly.
        % || p_j - <e, p_j>e ||^2 = <p_j, p_j> - 2<e, p_j><e, p_j> + <e, p_j><e, p_j><e, e> = <p_j, p_j> - <e, p_j>^2
        % the update factor becomes collectively (e'*P).^2
        colSquaredSums = colSquaredSums - (Q(:, k)'*P).^2;
        colSquaredSums(isColBasis) = 0;
    end    
end


%%
% Inner: fullQR()
%
% Inputs:
%   - P: NxN column-normalized co-occurrence matrix (column-stochastic)
%   - K: the number of basis vectors 
%
% Outputs:
%   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
%   - diagR: 1xK vector indicating the distance to the current subspace
%
% Remarks: 
%   - This function greedily selects K column vectors of P by using
%     Gram-Schmidt process, which is equivalent to column-pivoting.
%   - It does fully factorize P into QR.
%
function [S, diagR] = fullQR(P, K)
    % Perform full QR-factorization with column-pivoting.
    [~, R, S] = qr(P, 'vector');
    diagR = abs(diag(R))';
    
    % Extract values corresponding to only the given number of basis vectors.
    S = S(1:K);
    diagR = diagR(1:K);    
end


%%
% Inner: compressPCA(Cbar, dimension)
%
% Inputs:
%   - Cbar: NxN row-normalized co-occurrence matrix (column-stochastic)
%   - dimension: the dimension of new compressed representations
%
% Outputs:
%   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
%   - diagR: 1xK vector indicating the distance from centroid to each vertex
%
% Remarks:
%   - This function compresses given matrix (set of row vectors) via linear
%     projection using PCA.
%
function [S, diagR] = compressPCA(Cbar, dimension)
    % Perfrom PCA projection, and gets the row-rank approximation.
    [~, scores] = pca(Cbar);
    Cbar_proj = scores(:, 1:dimension);
    
    % Find the vertices of exact convex hull in the projected space.
    % Note that the number of basis anchors is not a user parameter, but a result of the discovered convex hull.
    vertices = convhull(Cbar_proj(:, 1), Cbar_proj(:, 2), Cbar_proj(:, 3));
    S = unique(vertices);
    K = numel(S);
    
    % Compute the diagR approximately by the distance from centroid to each vertex
    centroid = mean(Cbar_proj, 1);
    diagR = zeros(1, K);
    for k = 1:int32(K)
        diagR(k) = norm(Cbar_proj(S(k), :) - centroid, 2);
    end
    
    % Reorder S with respect to the magnitude of diagR element.
    [diagR, I] = sort(diagR, 'descend');
    S = S(I)';    
end


%%
% Inner: compressTSNE(Cbar, dimension)
%
% Inputs:
%   - Cbar: NxN row-normalized co-occurrence matrix (column-stochastic)
%   - dimension: the dimension of new compressed representations
%
% Outputs:
%   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
%   - diagR: 1xK vector indicating the distance from centroid to each vertex
%
% Remarks:
%   - This function compresses given matrix (set of row vectors) via
%     non-linear projection using t-SNE.
%
function [S, diagR] = compressTSNE(Cbar, dimension)
    % Perfrom the t-SNE projection, and gets the new representations.
    Cbar_proj = tsne(Cbar, [], dimension, 100);    

    % Find the vertices of exact convex hull in the projected space.
    % Note that the number of basis anchors is not a user parameter, but a result of the discovered convex hull.
    vertices = convhull(Cbar_proj(:, 1), Cbar_proj(:, 2), Cbar_proj(:, 3));
    S = unique(vertices);
    K = numel(S);
    
    % Compute the diagR approximately by the distance from centroid to each vertex.
    centroid = mean(Cbar_proj, 1);
    diagR = zeros(1, K);
    for k = 1:int32(K)
        diagR(k) = norm(Cbar_proj(S(k), :) - centroid, 2);
    end
    
    % Reorder S with respect to the magnitude of diagR element. 
    [diagR, I] = sort(diagR, 'descend');
    S = S(I)';    
end


%%
% Inner: influenceScore(P, K)
%
% Inputs:
%   - P: NxN column-normalized co-occurrence matrix (column-stochastic)
%   - K: the number of basis vectors 
%
% Outputs:
%   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
%   - diagR: 1xK vector indicating the influence.
%
% Remarks: 
%   - This function selects K most-influential column vectors of P based on the
%     influences that measures statistical leverage scores of each columns.
%   - The implementation is based on the CUR-decomposition paper.
%
function [S, diagR] = influenceScore(P, K)
    % Perform full singular-value decomposition.
    [~, ~, V] = svd(P);
    
    % Truncate the V matrix 
    V = V(:, 1:K);
    
    % Compute the statistical leverage scores of each columns in P.
    diagR = (V.^2)*ones(K, 1);
    
    % Pick the K column vectors with the largest influence.
    [diagR, S] = sort(diagR, 'descend');
    diagR = diagR(1:K)';
    S = S(1:K)';
end




%%
% TODO:
%


