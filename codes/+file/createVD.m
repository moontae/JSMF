%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Sungjun Cho 
% Modified: April, 2019
% Examples:
%   - [V, D] = file.createVD('docword.nips.txt', 'vocab.nips.txt', 10000, 'standard.stops', 3, 'nips_N-5000_comp', 5, 50);
%


%%
% Main: createVD()
% 
% Inputs:
%   - docword_filename: name of the input file containing bag-of-words
%   - vocab_filename: name of the dictionary file
%   - N: number of words in vocab
%   - stop_filename: name of the corpus file containing stop words
%   - min_objects: min. number of objects for effective training examples
%   - output_filename: name of the output file to write stat and dict
%   - min_tokens: min. number of tokens for effective training examples
%   - K: number of topics (rank of low-rank decomposition)
%   - T: number of power iterations to run within randomized eigendecomposition
%
% Outputs:
%   - V: N-by-K matrix
%   - D: K-by-K matrix
%   * V and D together approximate co-occurrence C = V*D*V'
%
% Remark:
%
function [V, D] = createVD(docword_filename, vocab_filename, N, stop_filename, min_objects, output_filename, min_tokens, K, T)
    % Set the default parameter.
    if nargin < 9
        T = 50;
    end
    if nargin < 7
        min_tokens = 5;
    end
    if nargin < 6
        output_filename = '';
    end
    if nargin < 5
        min_objects = 3;
    end        
    if nargin < 4
        stop_filename = '';
    end

    % Read bag of words.    
    [bows, ~] = file.readBows(docword_filename, vocab_filename, N, stop_filename, min_objects, output_filename);
    
    % Construct BOW matrix where each column represents a document.
    H = sparse(double(bows(:, 2)), double(bows(:, 1)), double(bows(:, 3)), N, double(bows(end, 1)));
         
    % Remove documents with less number of tokens than min_tokens.
    H(:, sum(H, 1) < min_tokens) = [];
        
    % Construct probe vectors.
    Z = randn(N, 2*K);
    
    % Compress the data.
    [V, D] = compression.randEig_Bows(Z, H, K, T);   
end