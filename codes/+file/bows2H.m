%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee & Sungjun Cho
% Examples:
%


%%
% Main: bows2H()
%
% Inputs:
%   - bows: X-by-3 bag-of-words matrix
%   - min_tokens: minimum number of tokens for each training example
%
% Outputs:
%   - H: NxM word-document matrix
%   - D1: Nx1 example frequency where D1_i = # of examples where object i occurs
%   - D2: NxN co-example frequency where D2_{ij} = # of examples where object i and j co-occurs
%
% Remark:
%   - This function converts bag-of-words to the full/sparse word-document matrix (double precision) 
%     and example/co-example frequencies by matrix operation as a whole.
%
function [H, D1, D2] = bows2H(bows, min_tokens)
    % Set the default parameter.
    if nargin < 2
        min_tokens = 5;
    end
    
    % Print out the initial status.
    fprintf('[file.bows2H] Start constructing sprase H...\n');
    startTime = tic;
    
    % Count the unique word indices.
    N = length(unique(bows(:, 2)));

    % Construct BOW matrix where each column represents a document.
    H = sparse(double(bows(:, 2)), double(bows(:, 1)), double(bows(:, 3)), N, double(bows(end, 1)));
         
    % Remove documents with less number of tokens than min_tokens.
    fprintf('+ Removing the documents based on min_tokens argument... \n');
    H(:, sum(H, 1) < min_tokens) = [];

    % Compute example and co-example frequencies if necessary.
    % Note that if vocaublary size is large, D2 can exceed the memory storage.
    if nargout >= 2
        D1 = double(sum(H > 0, 2));
    end
    if nargout >= 3
        U = sparse(double(H > 0));
        V = sparse(double(H == 1));
        D2 = U*U' - diag(sum(V.*V, 2));
    end    
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish constructing sparse H and D!\n');
    fprintf('  - The number of documents = %d\n', size(H, 2));
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);       
end




%%
% TODO:
%