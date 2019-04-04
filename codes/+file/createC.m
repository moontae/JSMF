%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [C, D1, D2, dict] = file.createC('../../jsmf-raw/dataset/real_raw/docword.nips.txt', '../../jsmf-raw/dataset/real_raw/vocab.nips.txt', '../../jsmf-raw/dataset/standard.stops', 5000, 3, 'nips_N-5000', 5);
%   - [C, D1, D2, ~] = file.createC('../../jsmf-raw/dataset/real_bows/nips_N-5000_train.bows');
%


%%
% Main: createC()
% 
% Inputs:
%   - bows_filename: name of the input file containing bag-of-words
%   - dict_filename: name of the dictionary file
%   - stop_filename: name of the corpus file containing stop words
%   - N: number of words in vocab
%   - min_objects: min. number of objects for effective training examples
%   - output_filename: name of the output file to write stat and dict
%   - min_tokens: min. number of tokens for effective training examples
%
% Outputs:
%   - C:  N-by-N full co-occurrence matrix
%   - D1: N-by-1 example frequency vector
%   - D2: N-by-N co-example frequency matrix
%   - dict: the original/curated dictonary of vocabulary
%
function [C, D1, D2, dict] = createC(bows_filename, dict_filename, stop_filename, N, min_objects, output_filename, min_tokens)
    % Set the default parameters.
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
        N = 0;
    end
    if nargin < 3
        stop_filename = '';
    end
    if nargin < 2
        dict_filename = '';
    end
        
    % Read bag of words.    
    dict = {};
    if (nargout >= 4) && ~isempty(dict_filename)
        [bows, dict] = file.readBows(bows_filename, dict_filename, stop_filename, N, min_objects, output_filename);
    else
        [bows, ~] = file.readBows(bows_filename, dict_filename, stop_filename, N, min_objects, output_filename);        
    end
    
    % Construct the full co-occurrence C.
    [C, D1, D2] = file.bows2C(bows, min_tokens);                
end




%%
% TODO:
%