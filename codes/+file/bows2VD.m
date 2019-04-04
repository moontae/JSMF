%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Sungjun Cho & Moontae Lee
% Examples:
%


%%
% Main: bows2VD()
%
% Inputs:
%   - bows: X-by-3 bag-of-words matrix
%   - K: number of topics (rank of low-rank decomposition)
%   - T: number of power iterations to run within randomized eigendecomposition
%   - min_tokens: min. number of tokens for effective training examples
%
% Outputs:
%   - V: N-by-K matrix
%   - D: K-by-K matrix (V and D together approximate co-occurrence C = V*D*V')
%   - D1: Nx1 example frequency where D1_i = # of examples where object i occurs
%   - D2: NxN co-example frequency where D2_{ij} = # of examples where object i and j co-occurs
%
% Remark:
%   - This function converts bag-of-objects to the compressed co-occurrence 
%     and example/co-example frequencies.
%

function [V, D, D1, D2] = bows2VD(bows, K, T, min_tokens)
   % Set the default parameter.
    if nargin < 4
        min_tokens = 5;
    end
    if nargin < 3
        T = 50;
    end
        
    % Print out the initial status.
    fprintf('[file.bows2VD] Start constructing compressed V and D...\n');
    startTime = tic;
    
    % Count the unique word indices.
    N = length(unique(bows(:, 2)));

    % Construct BOW matrix where each column represents a document.
    H = sparse(double(bows(:, 2)), double(bows(:, 1)), double(bows(:, 3)), N, double(bows(end, 1)));
         
    % Remove documents with less number of tokens than min_tokens.
    fprintf('+ Removing the documents based on min_tokens argument... \n');
    H(:, sum(H, 1) < min_tokens) = [];

    % Construct probe vectors.
    Z = randn(N, 2*K);
    
    % Compress the data.
    [V, D] = compression.randEig_Bows(Z, H, K, T);   
    
    % Compute example and co-example frequencies if necessary.
    % Note that if vocaublary size is large, D2 can exceed the memory storage.
    if nargout >= 3
        D1 = double(sum(H > 0, 2));
    end
    if nargout >= 4
        U = sparse(double(H > 0));
        V = sparse(double(H == 1));
        D2 = U*U' - diag(sum(V.*V, 2));
    end        
    elapsedTime = toc(startTime);
        
    % Print out the final status.
    fprintf('+ Finish constructing compressed V and D!\n');
    fprintf('  - The number of documents = %d\n', size(H, 2));
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);        
end



%%
% TODO:
%