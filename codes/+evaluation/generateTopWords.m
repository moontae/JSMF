%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%


%%
% Main: generateTopWords()
%
% Inputs:
%   - S: 1xK vector having the indices corresponding to K basis vectors
%   - B: NxK recovered object-cluster matrix
%   - L: the number of top contributing objects to print out
%   - dictionary_filename: the filename having the number-object mapping
%   - output_filename: the basename of output files
%
% Remarks: 
%   - This function generates both horizontal succinct version and vertical 
%     verbose version of each cluster with top contributing objects and
%     corresponding object-given-cluster probabilities.
%  
function generateTopWords(S, B, L, dictionary_filename, use_indices, output_filename)
    % Set the default option.
    if nargin < 6
        output_filename = 'topWords';
    end
    if isempty(S)
        % Print garbage basis object.
        S = ones(1, size(B, 2));
    end    
    
    % Print out the initial status.
    fprintf('Start generating top contributing objects...\n'); 
        
    % Read the mapping dictionary.
    dictionaryFile = fopen(dictionary_filename, 'r');
    dictionary = textscan(dictionaryFile, '%s');
    dictionary = dictionary{1}(use_indices);
    fclose(dictionaryFile);  
    fprintf('- Index-object mapping dictionary is properly loaded.\n');
    
    % Sort each group by the decreasing order of contributions an initialize.
    startTime = tic;
    [B_sorted, I] = sort(B, 1, 'descend');
    [N, K] = size(B);
    L = min(N, L);
    
    % Write the top M contributing members for each group.
    horOutputFile = fopen(strcat(output_filename, '.hor'), 'w');
    verOutputFile = fopen(strcat(output_filename, '.ver'), 'w');
    for k = 1:int32(K)
        fprintf(horOutputFile, '%20s\t', strcat('[', char(dictionary(S(k))), ']'));
        fprintf(verOutputFile, '[%s]\n', char(dictionary(S(k))));
        
        for l = 1:int32(L)
            fprintf(horOutputFile, ' %s', char(dictionary(I(l, k))));
            fprintf(verOutputFile, '\t%5d: %s (%.6f)\n', I(l, k), char(dictionary(I(l, k))), B_sorted(l, k));
        end
        fprintf(horOutputFile, '\n');        
        fprintf(verOutputFile, '\n');
    end
    fclose(horOutputFile);
    fclose(verOutputFile);
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish generating top contributing objects!\n');
    fprintf('  - Both horizontal/vertical files are generated!\n');
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);    
end




%%
% TODO:
%
