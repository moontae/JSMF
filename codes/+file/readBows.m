%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [bows, dict] = file.readBows('../../jsmf-raw/dataset/real_raw/docword.nips.txt',  '../../jsmf-raw/dataset/real_raw/vocab.nips.txt', '../../jsmf-raw/dataset/standard.stops', 5000, 5, '../dataset/nips_N-5000');
%   - bows = file.readBows('../../jsmf-raw/dataset/real_bows/nips_N-5000_train.bows');
%


%%
% Main: readBows()
%
% Inputs:
%   + bows_filename: the name of the input file containing bag-of-objects as a certain format
%     - 1st line: the number of training examples
%     - 2nd line: the maximum object number (some intermediate numbers might not actually exist)
%     - 3rd line: the number of rows below (not used)
%     - other lines: one space-delimited triplet (example #, object #, frequency count) per each line
%   + dict_filename: the name of the dictionary file of the corpus
%     - One object per each line
%   + stop_filename: the name of the corpus file containing stop objects
%     - One object per each line
%   - N: the size of effective objects (i.e., the size of vocabulary)
%   - min_objects: the minimum number of objects for each training example to be alive
%   - output_filename: the name of the output file reporting overall processing results
%
% Intermediates:
%   - M: the number of training examples 
%   - V: the maximum possible object number 
%   - activeObjects: the mapping from the reordered object numbers [1, N] --> originals [1, V]
%   - objectMap: the inverse map from originals [1, V] --> the reordered object numbers [1, N]
%
% Outputs:
%   - bows: matrix of filtered examples where each row contains (example #, object #, frequency)
%   - dict: the new dictionary mapping object number to actual objects
%
% Remarks: 
%   + This function reads UCI formatted Bag-of-words(objects) dataset.
%     - First, eliminate the stop objects.
%     - Second, prune out objects to the most effective N words based on tf-idfs.
%     - Third, remove the training examples that have less than minimum number of objects.
%     - Fourth, reorders the object numbers by assigning consecutive integers starting from 1.
%     - Last, it trims the dictionary according to the vocabulary size N.
%  
function [bows, dict] = readBows(bows_filename, dict_filename, stop_filename, N, min_objects, output_filename)
    % Print out the initial status.
    fprintf('[file.readBows] Start reading Bag-of-words dataset...\n');    
    startTime = tic;
    
    % Open the input file and read the statistics in the header.
    bowsFile = fopen(bows_filename, 'r');    
    inputLine = fgetl(bowsFile); M = sscanf(inputLine, '%d');
    inputLine = fgetl(bowsFile); V = sscanf(inputLine, '%d');
    fgetl(bowsFile); 
        
    % Step 0: Read the Bag-of-words(objects) content.
    bows = textscan(bowsFile, '%d %d %d');
    bows = [bows{1} bows{2} bows{3}];
    fclose(bowsFile);
    fprintf('- Dataset [%s] is parsed.\n', bows_filename);
    dict = {};
    if (nargin == 1) || (nargout == 1) || isempty(dict_filename)        
        return
    end       
    
    % Read the vocabulary dictionary if necessary.    
    if (nargin >= 2) && (nargout >= 2) && ~isempty(dict_filename)
        dict = readObjects(dict_filename);        
        fprintf('- Dictionary [%s] is loaded.\n', dict_filename);
        
        % Exit if only two arguments are received.
        if nargin == 2
            return
        end
    end
                
    % Step 1: Eliminate the stop objects if requested.
    if (nargin >= 3) && ~isempty(stop_filename)
        % Read the dictionary if it is not yet loaded.
        if isempty(dict)
            dict = readObjects(dict_filename);
        end
        
        % Read the stop objects.
        stop = readObjects(stop_filename);
                
        % Compute the overlap between the dictionary and the list of stop objects.
        stopObjects = find(ismember(dict, stop));        
        stopIndices = ismember(bows(:, 2), stopObjects);
        
        % Trim the bows by cropping out stop indices.
        bows = bows(~stopIndices, :);
        fprintf('- Stop objects are eliminated based on [%s].\n', stop_filename);
        
        % Exit if only three arguments are recdeived.
        if nargin == 3
            return 
        end
    end    
    
    % Check the pruning condition.
    if N <= 0
        % Exit immediately if N does not make sense.
        fprintf('* No pruning has been done!\n');
        return
    end   
           
    % Step 2: Compute the indices where each new training example starts.        
    [~, endRows, ~] = intersect(bows(:, 1), 1:max(bows(:, 1)));
    M = numel(endRows);
    endRows = [endRows; size(bows, 1)+1];    
    
    % Compute the term-frequencies, document-frequencies, and inverse document-frequencies.
    tfs = zeros(V, 1, class(bows));
    dfs = zeros(V, 1, class(bows));
    for m = 1:M
        % Extract the interval of indices corresponding to each training example.
        startRow = endRows(m);
        endRow = endRows(m+1)-1;
        
        % Read the list of object numbers and corresponding counts.
        objects = bows(startRow:endRow, 2);
        counts = bows(startRow:endRow, 3);
        
        % Accumulate the term-frequencies by their count occurrences.
        tfs(objects) = tfs(objects) + counts;
        
        % Increase the document-frequency of each object by 1.
        dfs(objects) = dfs(objects) + 1;
    end
    tfs = cast(tfs, 'double');
    dfs = cast(dfs, 'double');          
        
    % Note that we at least prune all the object which shows up more than 50% of documents. 
    % This is our document-frequency cut-off to be included in the effective vocabulary.    
    % Option 1: Integer flooring on df scores.
    idfs = log(floor(M ./ dfs)); 
    % Option 2: Just remove objects more than half of documents.
    %idfs = log(M ./ dfs);    
    %idfs(dfs > M/2) = 0; 
    
    % Evaluate the tf-idf scores.
    tfIdfs = tfs .* idfs;
    
    % Sort the tf-idf scores by the decrasing order and filter out NaN.
    % Note that V is not the vocabulary size, but the maximum possible 
    % object number. Some numbers could be unused, making tf-idf as 0/0.
    nanIndices = isnan(tfIdfs);
    tfIdfs(nanIndices) = 0;
    [~, indices] = sort(tfIdfs, 'descend');
    N = min(N, V - sum(nanIndices));
       
    % Discard every bag-of-word entries with an irrelevant object.
    % Note that actual number of objects = V - non-existing objects
    irrelevantObjects = indices(N+1:end);
    irrelevantIndices = ismember(bows(:, 2), irrelevantObjects);
    bows = bows(~irrelevantIndices, :);
    fprintf('- Objects are pruned.\n');
    
    % Step 3: Remove training examples having less than the minimum number of objects.
    % Recalculate the indices where each new training example starts.
    fprintf('+ Removing the documents based on min_objects argument...\n');
    [~, endRows, ~] = intersect(bows(:, 1), 1:max(bows(:, 1)));    
    endRows = [endRows; size(bows, 1)+1];   
    
    % Compute the number of differnt objects in each examples.
    % Note that we consider how many different type of objects, not the number of object occurrences.
    numObjects = diff(endRows);    
    
    % Filter out the documents with less than min_objects threshold.
    removeExamples = find(numObjects < min_objects);    
    activeIndices = true(size(bows, 1), 1);
    for idx = 1:numel(removeExamples)
        m = removeExamples(idx);
        activeIndices(endRows(m):endRows(m+1)-1) = false;        
        fprintf('  - The document %d with only %d objects will be ignored!\n', bows(endRows(m), 1), numObjects(m));
    end    
    bows = bows(activeIndices, :);
            
    % Step 4: Reorder the remaining objects by assigning new consecutive object numbers.
    activeObjects = unique(bows(:, 2));
    N = numel(activeObjects);
    objectMap = zeros(1, V);
    objectMap(indices(1:N)) = 1:N;
    bows(:, 2) = objectMap(bows(:, 2));    
    fprintf('- Reordering object numbers is done.\n');
    elapsedTime = toc(startTime);
            
    % Save the statistics if output is specified.
    if (nargin >= 6) && ~isempty(output_filename)
        outputFile = fopen(strcat(output_filename, '.stat'), 'w');
        for v = 1:int32(V)
            object = indices(v);           
            fprintf(outputFile, '%6d\t%8d\t%-20s\t%.6f\n', v, object, char(dict(object)), tfIdfs(object));            
        end
        fclose(outputFile);
        fprintf('- Pruning statistics is generated.\n');
    end
       
    % Save the curated N objects as a new vocabulary dictionary.
    dict = dict(indices(1:N));
    if (nargin >= 6) && ~isempty(output_filename)
        outputFile = fopen(strcat(output_filename, '.dict'), 'w');
        for n = 1:int32(N)
            fprintf(outputFile, '%s\n', char(dict(n)));
        end
        fclose(outputFile);
        fprintf('- Curated dictionary file is generated.\n');
    end        
    
    % Print out the final status.
    fprintf('+ Finish reading Bag-of-words dataset!\n');
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);       
end

    
%%
% Inner: readObjects(filename)
%
% Inputs:
%   - filename: text file that includes one object/word per each line.
%
% Outputs:
%   - words: cell where each element contains corresponding object/word.
%
function objects = readObjects(filename)        
    % Open the file and read each object/word from every line.
    file = fopen(filename, 'r');    
    objects = textscan(file, '%s');
    objects = objects{1};
    fclose(file);    
end
    
    


%%
% TODO:
%