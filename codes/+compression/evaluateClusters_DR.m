%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   


%%
% Main: evaluateClusters_DR()
%
% Inputs:
%   - S: 1xK column vector having the basis indices
%   - Btilde: KxN cluster-object matrix where Btilde_{kn} = p(Z=k | X=n)
%   - Y: NxK rectified and compressed co-occurrence
%   - E: NxN sparse correction matrix
%
% Outputs:
%   - values: two average recovery errors on full and half vocabulary
%   - stdevs: two standard deviations on full and half vocabulary
%
% Remarks: 
%   - This function evaluates the approximate recovery error by iteration
%     without random projections only given given the ENN rectification.
%   - It measures the errors on both the full vocabulary and the first half 
%     of the vocabulary.
%
function [fullValue, partialValue] = evaluateClusters_DR(S, Btilde, Y, E, ratio)
    N = size(Btilde, 2);
    N0 = round(N * ratio);

    % Compute the column-sum of C, and prepare a corresponding diagonal matrix.    
    d = sum(Y)*Y' + sum(E);            
    
    % Compute row-normalized Y.
    Ybar = bsxfun(@ldivide, d', Y);
    
    % Compute the several useful intermediates.    
    % Ebar = E*inv --> Ebart = invD*E^T = invD*E.
    Ebart = (E./d)';
    X = Ybar - Btilde'*Ybar(S, :);    
    
    % Compute the full and half recovery errors.
    fullValue = sum(recoveryError_slice(S, Btilde, X, Y, Ebart))/N;
    partialValue = sum(recoveryError_slice(S, Btilde(:, 1:N0), X(1:N0, :), Y, Ebart))/N0;
end

    
function errors = recoveryError_slice(S, Btilde, X, Y, Ebart, capacity)    
    % Set the default memory capacity.
    if nargin < 6
        capacity = 1e8;
    end    
    
    % Split large set of objects into multiple slices.    
    N = size(Btilde, 2);
    numSlices = min(N, floor(N*N / capacity));
    if numSlices == 0
        sliceSizes = N;
    else
        sliceSizes = [floor(N / numSlices)*ones(1, numSlices), mod(N, numSlices)];
    end    
    objectSlices = mat2cell(1:N, 1, sliceSizes);    
    
    % For each slice,
    numSlices = length(objectSlices);
    errors = zeros(1, numSlices);
    for s = 1:numSlices
        % Evaluate recovery error just for the current slice.
        objects = objectSlices{s}';        
        residual = X(objects, :)*Y' + Ebart(objects, :) - Btilde(:, objects)'*Ebart(S, :);
        errors(s) = sum(sqrt(sum(abs(residual).^2, 2)));
    end     
end
    



%%
% TODO:
%     (old code for computing half/full residuals without slicing)
%   
%     % Compute the half residuals without slicing.
%     residuals = X(1:N0, :)*Y' + Ebart(1:N0, :) - Btilde(:, 1:N0)'*Ebart(S, :);
%     errors = sqrt(sum(abs(residuals).^2, 2));    
%     fullValue = mean(errors);          
%
%     % Compute the full residuals without slicing
%     residuals = X*Y' + Ebart - Btilde'*Ebart(S, :);
%     errors = sqrt(sum(abs(residuals).^2, 2));    
%     fullValue = mean(errors);     
%
    
    
    
    






