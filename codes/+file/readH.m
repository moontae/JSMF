%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - H = file.readH('../../jsmf-raw/dataset/real_bows/nips_N-5000_train.bows');


%%
% Main: readH()
%
% Remarks:
%   - This function reads UCI formatted Bag-of-words(objects) dataset sequentially for each document,
%     generating a full word-document matrix H with a specific type such as int32.
%
function H = readH(bows_filename, type)
    % Set the default option.
    if nargin < 2
        type = 'double';
    end

    % Print out the initial status
    fprintf('[file.readH] Start reading H from Bag-of-words dataset...\n');    
    startTime = tic;
        
    % Read the statistics in the header. 
    bowsFile = fopen(bows_filename, 'r');
    inputLine = fgetl(bowsFile); M = sscanf(inputLine, '%d');
    inputLine = fgetl(bowsFile); N = sscanf(inputLine, '%d');
    fgetl(bowsFile); 
        
    % Read the bag-of-objects starting from the next lines.
    bows = textscan(bowsFile, '%d %d %d');
    bows = [bows{1} bows{2} bows{3}];
    fclose(bowsFile);
    fprintf('+ Dataset [%s] is parsed.\n', bows_filename);
    fprintf('  - Maximum counts in this dataset = %d\n', max(bows(:, 3)));
    fprintf('  - Ensure that [%s] can cover the range of the maximum counts!\n', type);          

    % Step 2: Compute the indices where each new training example ends.   
    [~, endRows, ~] = intersect(bows(:, 1), 1:max(bows(:, 1)));
    M = numel(endRows);    
    endRows = [endRows; size(bows, 1)+1];       
        
    % For each document,
    H = zeros(N, M, type);    
    fprintf('+ Creating a word-document matrix...\n');
    for m = 1:M
        % Print out the progress.
        if mod(m, 5000) == 0
            fprintf('  - %d-th document...\n', m);
        end
        
        % Extract the interval of indices corresponding to each training example.
        startRow = endRows(m);
        endRow = endRows(m+1)-1;
                
        % Assign the word-count statistics for the current document.
        rows = startRow:endRow;
        words = bows(rows, 2);
        counts = bows(rows, 3);
        H(words, m) = counts;        
    end
    elapsedTime = toc(startTime);
    
    % Print out the final status.    
    fprintf('+ Finish reading H!\n');
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);     
end




%%
% TODO:
%
