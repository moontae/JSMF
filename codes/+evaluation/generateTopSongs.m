%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%


%%
% Main: generateTopSongs()
%
% Inputs:
%   - S: 1xK vector having the indices corresponding to K basis vectors
%   - B: NxK recovered object-cluster matrix
%   - L: the number of top contributing objects to print out
%   - dict_filename: the filename having the number-object mapping
%   - use_indices: list of effective word indices
%   - output_filename: the basename of output files
%
% Remarks: 
%   - This function generates both horizontal succinct version and vertical 
%     verbose version of each cluster with top contributing objects and
%     corresponding object-given-cluster probabilities.
%  
function generateTopSongs(S, B, L, dict_filename, use_indices, output_filename)
    % Set the default option.
    if nargin < 6
        output_filename = 'topSongs';
    end
    if isempty(S)
        % Print garbage basis object.
        S = ones(1, size(B, 2));
    end    
    
    % Print out the initial status.
    fprintf('[evaluation.generateTopSongs] Start generating top contributing songs...\n'); 
        
    % Read the mapping dictionary.
    dictFile = fopen(dict_filename, 'r');
    dict = textscan(dictFile, '%s %s', 'delimiter', '\t');
    titleDict = dict{1}(use_indices);
    artistDict = dict{2}(use_indices);
    fclose(dictFile);  
    fprintf('- Index-song-mapping is properly loaded.\n');
    
    % sort each group by the decreasing order of contributions and initialize
    startTime = tic;
    [B_sorted, I] = sort(B, 1, 'descend');
    [N, K] = size(B);
    L = min(N, L);
    
    % Write the top M contributing members for each group.
    outputFile = fopen(strcat(output_filename, '.ver'), 'w');
    for k = 1:int32(K)
        fprintf(outputFile, '[%s <%s>]\n', char(titleDict(S(k))), char(artistDict(S(k))));
        
        for l = 1:int32(L)
            id = I(l, k);
            fprintf(outputFile, '\t%5d: %s <%s> (%.6f)\n', id, char(titleDict(id)), char(artistDict(id)), B_sorted(l, k));
        end
        fprintf(outputFile, '\n');
    end
    fclose(outputFile);
    elapsedTime = toc(startTime);
    
    % print out the final status
    fprintf('+ Finish generating top contributing movies...\n');
    fprintf('  - Only vertical files are generated.\n');
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);     
end




%%
% TODO:
%