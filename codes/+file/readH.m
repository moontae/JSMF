%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:


%%
% Main: readH()
%
% Remarks:
%   - This function parses word-document matrix from the given UCI bag-of-words input file.
%   - Document numbers are re-algined as continuous increasing numbers starting from 1.
%
function H = readH(input_filename, type)
    % Set the default option.
    if nargin < 2
        type = 'double';
    end

    % Print out the initial status
    fprintf('Start reading the term-document matrix H...\n');
    fprintf('- Reading %s...\n', input_filename);
    tic;
        
    % Read the statistics in the header. 
    inputFile = fopen(input_filename, 'r');
    inputLine = fgetl(inputFile); M = sscanf(inputLine, '%d');
    inputLine = fgetl(inputFile); N = sscanf(inputLine, '%d');
    fgetl(inputFile); 
        
    % Read the bag-of-objects starting from the next lines.
    bows = textscan(inputFile, '%d %d %d');
    bows = [bows{1} bows{2} bows{3}];
    fprintf('- Maximum counts in this dataset = %d\n', max(bows(:, 3)));
    fprintf('- Ensure that [%s] can cover the range of the maximum counts!\n', type);    
    fclose(inputFile);
        
    % Compute starting rows for all documents and prepares the word-document matrix.
    [~, startRows] = intersect(bows(:, 1), 1:max(bows(:, 1)));
    M = int32(numel(startRows));
    H = zeros(N, M, type);    
    
    % Pad the last row to cover the rows for the last document.
    startRows = [startRows; size(bows, 1) + 1];
    
    % For each document,
    fprintf('+ Progress...\n');
    for m = 1:M
        % Get the start end end rows.
        startRow = int32(startRows(m));
        endRow = int32(startRows(m+1) - 1);
                
        % Print out the progress.
        if mod(m, 5000) == 0
            fprintf('  - %d-th document...\n', m);
        end
        
        % Assign the word-count statistics for the current document.
        rows = startRow:endRow;
        words = bows(rows, 2);
        counts = bows(rows, 3);
        H(words, m) = counts;        
    end
end




%%
% TODO:
%
