%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%


%%
% Main: generateTopMovies()
%
% Inputs:
%   - S: 1xK vector having the indices corresponding to K basis vectors
%   - B: NxK recovered object-cluster matrix
%   - L: the number of top contributing objects to print out
%   - dict_filename: the filename having the number-object mapping
%   - output_filename: the basename of output files
%
% Remarks: 
%   - This function generates both horizontal succinct version and vertical 
%     verbose version of each cluster with top contributing objects and
%     corresponding object-given-cluster probabilities.
%  
function generateTopMovies(S, B, L, dict_filename, use_indices, output_filename)
    % Set the default option.
    if nargin < 6
        output_filename = 'topMovies';
    end
    if isempty(S)
        % Print garbage basis object.
        S = ones(1, size(B, 2));
    end    
    
    % Print out the initial status.
    fprintf('[evaluation.generateTopMovies] Start generating top contributing movies...\n'); 
        
    % Read the mapping dictionary.
    dictFile = fopen(dict_filename, 'r');
    dict = textscan(dictFile, '%s %d %d', 'delimiter', '\t');
    titleDict = dict{1}(use_indices);
    yearDict  = dict{2}(use_indices);
    genreDict = dict{3}(use_indices);
    fclose(dictFile);  
    fprintf('- Index-movie mapping dictionary is properly loaded.\n');
    
    % Sort each group by the decreasing order of contributions and initialize.
    startTime = tic;
    [B_sorted, I] = sort(B, 1, 'descend');
    [N, K] = size(B);
    L = min(N, L);
    
    % Write the top M contributing members for each group.
    outputFile = fopen(strcat(output_filename, '.ver'), 'w');
    for k = 1:int32(K)
        fprintf(outputFile, '[%s <%d> <%s>]\n', char(titleDict(S(k))), yearDict(S(k)), getStringGenre(genreDict(S(k))));
        
        for l = 1:int32(L)
            id = I(l, k);
            fprintf(outputFile, '\t%5d: %s <%d> <%s> (%.6f)\n', id, char(titleDict(id)), yearDict(id), getStringGenre(genreDict(id)), B_sorted(l, k));
        end
        fprintf(outputFile, '\n');
    end
    fclose(outputFile);
    elapsedTime = toc(startTime);
    
    % Print out the final status.
    fprintf('+ Finish generating top contributing movies!\n');
    fprintf('  - Only vertical files are generated.\n');
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);    
end


%%
% Inner: getStringGenre()
%
% Inputs:
%   - genre: integer number indicating genre mixture
%
% Outputs:
%   - strinGenre: string genre concatenated by the bar
%
% Remarks:
%   - This function converts integer genre into string version.
%
function stringGenre = getStringGenre(genre)
    stringGenre = '';
    
    if mod(genre, 2) == 0
        stringGenre = strcat(stringGenre, 'Action|');
    end
    if mod(genre, 3) == 0
        stringGenre = strcat(stringGenre, 'Adventure|');
    end
    if mod(genre, 5) == 0
        stringGenre = strcat(stringGenre, 'Animation|');
    end
    if mod(genre, 7) == 0
        stringGenre = strcat(stringGenre, 'Children|');
    end
    if mod(genre, 11) == 0
        stringGenre = strcat(stringGenre, 'Comedy|');
    end
    if mod(genre, 13) == 0
        stringGenre = strcat(stringGenre, 'Crime|');
    end
    if mod(genre, 17) == 0
        stringGenre = strcat(stringGenre, 'Documentary|');
    end
    if mod(genre, 19) == 0
        stringGenre = strcat(stringGenre, 'Drama|');
    end
    if mod(genre, 23) == 0
        stringGenre = strcat(stringGenre, 'Fantasy|');
    end
    if mod(genre, 29) == 0
        stringGenre = strcat(stringGenre, 'Film-Noir|');
    end
    if mod(genre, 31) == 0
        stringGenre = strcat(stringGenre, 'Horror|');
    end
    if mod(genre, 37) == 0
        stringGenre = strcat(stringGenre, 'Musical|');
    end
    if mod(genre, 41) == 0
        stringGenre = strcat(stringGenre, 'Mystery|');
    end
    if mod(genre, 43) == 0
        stringGenre = strcat(stringGenre, 'Romance|');
    end
    if mod(genre, 47) == 0
        stringGenre = strcat(stringGenre, 'Sci-Fi|');
    end
    if mod(genre, 53) == 0
        stringGenre = strcat(stringGenre, 'Thriller|');
    end
    if mod(genre, 59) == 0
        stringGenre = strcat(stringGenre, 'War|');
    end
    if mod(genre, 61) == 0
        stringGenre = strcat(stringGenre, 'Western|');
    end
    
    stringGenre = stringGenre(1:length(stringGenre) - 1);
end




%%
% TODO:
%
