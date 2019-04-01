%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%   - [bows, dictionary] = readBows('../../dataset/real_bows/docword.nips.txt',  '../../dataset/real_bows/vocab.nips.txt', '../../dataset/standard.stops', 5000, 5, '../../dataset/nips_N-5000');
%


%%
% Main: readBows()
%
% Inputs:
%   + docword_filename: the name of the input file containing bag-of-objects as a certain format
%     - 1st line: the number of training examples
%     - 2nd line: the maximum object number (some intermediate numbers might not actually exist)
%     - 3rd line: the number of rows below (not used)
%     - other lines: one space-delimited triplet (example #, object #, frequency count) per each line
%   + dictionary_filename: the name of the dictionary file of the corpus
%     - One object per each line
%   - N: the size of effective objects (i.e., the size of vocabulary)
%   + stop_filename: the name of the corpus file containing stop objects
%     - One object per each line
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
%   - dictionary: the new dictionary mapping object number to actual objects
%
% Remarks: 
%   + This function reads UCI formatted bag-of-objects dataset.
%     - First, eliminate the stop objects.
%     - Second, prune out objects to the most effective N words based on tf-idfs.
%     - Third, remove the training examples that have less than minimum number of objects.
%     - Fourth, reorders the object numbers by assigning consecutive integers starting from 1.
%     - Last, it trims the dictionary according to the vocabulary size N.
%  
function [bows, dictionary] = readBows(docword_filename, vocab_filename, N, stop_filename, min_objects, output_filename)
    % Print out the initial status.
    fprintf('Start reading Bag-of-Words dataset...\n');
    startTime = tic;
    
    % Open the input file and read the statistics in the header.
    inputFile = fopen(docword_filename, 'r');    
    inputLine = fgetl(inputFile); M = sscanf(inputLine, '%d');
    inputLine = fgetl(inputFile); V = sscanf(inputLine, '%d');
    fgetl(inputFile); 
        
    % Step 0: Read the rest bag-of-words(objects).
    bows = textscan(inputFile, '%d %d %d');
    bows = [bows{1} bows{2} bows{3}];
    fclose(inputFile);
    fprintf('- The dataset is parsed.\n');
        
    % Check the condition for vocabulary pruning.
    if N <= 0
        % Never touch and just read as it is.
        vocabFile = fopen(vocab_filename, 'r');    
        dictionary = textscan(vocabFile, '%s');
        dictionary = dictionary{1};
        fclose(vocabFile);
        return
    end   
    
    % Step 1: Eliminate the stop objects and read the dictionary map.    
    if strcmp(stop_filename, '')
        % Only read the dictionary if no elimination is required.
        vocabFile = fopen(vocab_filename, 'r');    
        dictionary = textscan(vocabFile, '%s');
        dictionary = dictionary{1};
        fclose(vocabFile);
    else
        % Read the dictionary and eliminate the stop objects.
        [stopObjects, dictionary] = getStops(stop_filename, vocab_filename);
        stopIndices = ismember(bows(:, 2), stopObjects);
        bows = bows(~stopIndices, :);
        fprintf('- Stop objects are eliminated.\n');
    end
        
    % Step 2: Compute the indices where each new training example starts.
    [~, endRows, ~] = intersect(bows(:, 1), 1:M);
    endRows = [endRows; size(bows, 1)+1];    
    
    % Compute the term-frequencies, document-frequencies, and inverse document-frequencies.
    tfs = zeros(V, 1, class(bows));
    dfs = zeros(V, 1, class(bows));
    for m = 1:length(endRows)-1
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
    [~, endRows, ~] = intersect(bows(:, 1), 1:M);
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
    elapsedTime = toc(startTime);
    fprintf('- Reordering object numbers is done.\n');
            
    % Print out the statistics.
    if (nargin >= 6) && ~strcmp(output_filename, '')
        outputFile = fopen(strcat(output_filename, '.stat'), 'w');
        for v = 1:int32(V)
            object = indices(v);           
            fprintf(outputFile, '%6d\t%8d\t%-20s\t%.6f\n', v, object, char(dictionary(object)), tfIdfs(object));            
        end
        fclose(outputFile);
        fprintf('- Pruning statistics is generated.\n');
    end
       
    % Save the effective N objects as a vocabulary dictionary.
    dictionary = dictionary(indices(1:N));
    if (nargin >= 6) && ~strcmp(output_filename, '')
        outputFile = fopen(strcat(output_filename, '.dict'), 'w');
        for n = 1:int32(N)
            fprintf(outputFile, '%s\n', char(dictionary(n)));
        end
        fclose(outputFile);
        fprintf('- Dictionary file is generated.\n');
    end        
    
    % Print out the final status.
    fprintf('+ Finish reading bag-of-objects\n');
    fprintf('  - Elapsed seconds = %.4f\n\n', elapsedTime);       
end

    
%%
% Inner: getStops()
%
% Inputs:
%   + stop_filename: the name of the corpus file containing stop objects
%     - One object per each line
%   + dictionary_filename: the name of the dictionary file of the corpus
%     - One object per each line
%
% Outputs:
%   - stopObjects: the object numbers corresponding to stops objects.
%   - dictionary: the dictionary mapping object number to actual objects.
%
function [stopObjects, dictionary] = getStops(stop_filename, vocab_filename)
    % open the stop file and read every line
    stopFile = fopen(stop_filename, 'r');    
    stopWords = textscan(stopFile, '%s');
    stopWords = stopWords{1};
    fclose(stopFile);
        
    % open the dictionary file and read every line
    vocabFile = fopen(vocab_filename, 'r');    
    dictionary = textscan(vocabFile, '%s');
    dictionary = dictionary{1};
    fclose(vocabFile);
          
    % compute the object numbers corresponding to stop objects
    stopObjects = find(ismember(dictionary, stopWords));
end
    
    


%%
% TODO:
%