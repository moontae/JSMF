%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%   - [C, D1, D2] = file.createC('docword.nips.txt', 'vocab.nips.txt', 5000, 'standard.stops', 3, 'nips_N-5000', 5);
%


%%
% Main: createC()
% 
% Inputs:
%   - docword_filename: name of the input file containing bag-of-words
%   - vocab_filename: name of the dictionary file
%   - N: number of words in vocab
%   - stop_filename: name of the corpus file containing stop words
%   - min_objects: min. number of objects for effective training examples
%   - output_filename: name of the output file to write stat and dict
%   - min_tokens: min. number of tokens for effective training examples
%
% Outputs:
%   - C:  N-by-N full co-occurrence matrix
%   - D1: N-by-1 example frequency vector
%   - D2: N-by-N co-example frequency matrix
%
% Remark:
%
function [C, D1, D2] = createC(docword_filename, vocab_filename, N, stop_filename, min_objects, output_filename, min_tokens)
    % Set the default parameter.
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
    
    % Construct the full co-occurrence C.
    [C, D1, D2] = file.convertBows(bows, min_tokens);            
end