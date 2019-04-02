%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Sungjun Cho 
% Modified: April, 2019
% Examples:
%


%%
% Main: makeDataset()
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
%   + If called with two outputs
%      - V: N-by-K matrix
%      - D: K-by-K matrix
%      - V and D together approximate co-occurrence C = V*D*V'
%   + If called with three outputs
%      - C: N-by-N full co-occurrence matrix
%      - D1: N-by-1 example frequency vector
%      - D2: N-by-N co-example frequency matrix
%
% Example:
%   <If you want to pre-process bows and construct full C>
%   - [C, D1, D2] = file.makeDataset('docword.nips.txt', 'vocab.nips.txt', 5000, 'standard.stops', 3, 'nips_N-5000', 5);
%
%   <If you want to construct compressed data>
%   - [V, D] = file.makeDataset('docword.nips.txt', 'vocab.nips.txt', 10000, 'standard.stops', 3, 'nips_N-5000_comp', 5, 50);
%
function [out1, out2, out3] = makeDataset(docword_filename, vocab_filename, N, stop_filename, min_objects, output_filename, min_tokens, K, T)
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
    if nargin == 3
        [bows, ~] = file.readBows(docword_filename, vocab_filename, N);
    else
        [bows, ~] = file.readBows(docword_filename, vocab_filename, N, stop_filename, min_objects, output_filename);
    end
    
    % If asked for compressed statistics,
    if (nargout == 2) && (nargin >= 7)
        % Construct BOW matrix where each column represents a document.
        H = sparse(double(bows(:, 2)), double(bows(:, 1)), double(bows(:, 3)), N, double(bows(end, 1)));
         
        % Remove documents with less number of tokens than min_tokens.
        H(:, sum(H, 1) < min_tokens) = [];
        
        % Construct probe vectors.
        Z = randn(N, 2*K);
        [out1, out2] = compression.randEig_Bows(Z, H, K, T);
        
    % If asked for full co-occurrence statistics,
    elseif nargout == 3         
        % Construct the full co-occurrence C.
        [out1, out2, out3] = file.constructCD(bows, min_tokens);        
    end    
end