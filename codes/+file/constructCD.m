%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee & Sungjun Cho
% Modified: April, 2019
% Examples:
%


%%
% Main: constructCD()
%
% Inputs:
%   - bows: X-by-3 bag-of-words matrix
%   - min_tokens: min. number of tokens for effective training examples
%
% Outputs:
%   - C: NxN dense joint-stochastic co-occurrence matrix 
%   - D1: Nx1 example frequency where D1_i = # of examples where object i occurs
%   - D2: NxN co-example frequency where D2_{ij} = # of examples where object i and j co-occurs
%
function [C, D1, D2] = constructCD(bows, min_tokens)
    % Print out the initial status.
    fprintf('Start constructing dense C and D...\n');
    
    % Recompute the size of vocabulary by counting the unique elements in the word numbers.
    N = length(unique(bows(:, 2)));
    
    % Find the row numbers where each training example ends. 
    [~, endRows, ~] = intersect(bows(:, 1), 1:max(bows(:, 1)));
    
    % Recompute the number of documents.    
    % Note that some document may be disappeared after pruning.
    M = numel(endRows);         
    endRows = [endRows; size(bows, 1) + 1];    
    
    % Compute co-occurrence and example/co-example frequencies for each training example.
    fprintf('+ Removing the documents based on min_tokens argument... \n');
    startTime = tic;    
    C = zeros(N, N);
    D1 = zeros(N, 1);
    D2 = zeros(N, N);
    for m = 1:int32(M)                
        % Determine the start and end rows for this document.
        startRow = endRows(m);
        endRow = endRows(m+1) - 1;
        objects = cast(bows(startRow:endRow, 2), 'double');
        counts = cast(bows(startRow:endRow, 3), 'double');
        
        % Skip the degenerate case when the document contains only one word with a single occurrence.
        % Note that it does not happen if min_object threshold is larger 1 when reading bows.    
        numObjects = length(objects);
        numTokens = sum(counts);
        if (numObjects == 1) && (numTokens == 1)
            % Note that 1*1 - 1 = 0 causes dividing by zero, yielding NaN.
            continue;
        end        
        
        % Skip the current example with less than minimum counts        
        if numTokens < min_tokens
            fprintf('  - The document %d with only %d tokens will be ignored!\n', m, numTokens);
            continue;
        end        
        
        % Accumulate correponsding counts to co-occurrence and example/co-example frequencies.
        % Note that co-example frequency for an object can exist only when the object occurs more than once.
        normalizer = numTokens*(numTokens - 1);
        C(objects, objects) = C(objects, objects) + (counts*counts' - diag(counts))/normalizer;
        D1(objects) = D1(objects) + 1;        
        D2(objects, objects) = D2(objects, objects) + 1 - diag(counts == 1);        
    end
    
    % Ensure the overall sum is equal to 1.0.
    entrySum = sum(sum(C));
    if (entrySum ~= M)
        C = C ./ entrySum; 
    end    
    elapsedTime = toc(startTime);
    
    % Print out the final status
    fprintf('+ Finish constructing C and D!\n');
    fprintf('  - The sum of all entries = %.6f\n', entrySum / M);
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);       
end