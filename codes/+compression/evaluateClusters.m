%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%   - [value, stdev] = evaluateClusters('all', A1, B1, S1, A1tilde, C1bar, C1_rowSums, C1, D, D2, 1);
%   - [value, stdev] = evaluateClusters('allButD', A, B, S, Btilde, Cbar, C_rowSums, C, 1);
%   - [value, stdev] = evaluateClusters('allButS', A, B, Btilde, C_rowSums, C, D, D2, 0);
%   - evaluateClusters('recoveryError', Cbar, S, Btilde);
%


%%
% Main: evaluateClusters()
%
% Remark: This function is a wrapper to call the inner functions given below.
%
function [value, stdev] = evaluateClusters(varargin)    
    if nargout == 2
        [value, stdev] = feval(varargin{1}, varargin{2:end});    
    else
        value = feval(varargin{1}, varargin{2:end});    
    end
end


%%
% Inner: allForComp()
%
% Remark: 
%   - This function evaluates every possible metric based on the
%   compressed co-occurrence information.
%
function [value, stdev] = allGivenH(S, B, A, Btilde, H, C_rowSums, withTitle)
    % Decide whether or not printing out metric titles.
    if nargin < 7
        withTitle = 0;
    end
    
    % Return only the title information.
    if withTitle == -1
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');  
        return
    end
    
    % Setup the dissimilarity measure to use.
    clusterDissimilarity_soft = @clusterDissimilarity_symKL;
    %clusterDissimilarity_soft = @clusterDissimilarity_cos;
    %clusterDissimilarity_soft = @clusterDissimilarity_Fisher;
    
    % Measure all metrics.
    [RE, RE_std]   = compression.evaluateClusters_RE(S, Btilde, H); 
    DL             = distributionLegality(A);
    MV             = marginalValidity(B, A, Btilde, C_rowSums);
    [AE1, AE2]     = compression.evaluateClusters_AE(B, A, H);
    [DD, DD_std]   = diagonalDominancy(A);
    
    [NE, NE_std]   = normalizedEntropy(Btilde);
    [CP, CP_std]   = clusterSparsity(B);
    [CS, CS_std]   = clusterSpecificity(B, C_rowSums);    
    [CDh, CDh_std] = clusterDissimilarity_hard(B, 20);
    [CDs, CDs_std] = clusterDissimilarity_soft(B);
    
    % Does not measure coherence metrics.
    CC = 0;        
    CC_std = 0;
    
    [BRh, BRh_std] = basisRank_hard(B, S);
    [BRs, BRs_std] = basisRank_soft(B, S);
    [BQh, BQh_std] = basisQuality_hard(B, S);
    [BQs, BQs_std] = basisQuality_soft(B, S);
        
    % Print out the results.
    value = sprintf('%14.6f %14.4f %14.4f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14.4f %14.4f %14.2f %14.6f %14.6f', RE, DL, MV, AE1, AE2, DD, NE, CS, CDh, CDs, CC, BRh, BRs, BQh, BQs, CP);
    stdev = sprintf('%14.6f %14s %14s %14s %14s %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14.4f %14.4f %14.2f %14.6f %14.6f', RE_std, 'NA', 'NA', 'NA', 'NA', DD_std, NE_std, CS_std, CDh_std, CDs_std, CC_std, BRh_std, BRs_std, BQh_std, BQs_std, CP_std);
    if withTitle
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', value);        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', stdev);        
    end        
end


%%
% Inner: allForComp()
%
% Remark: 
%   - This function evaluates every possible metric based on the
%   compressed co-occurrence information.
%
function [value, stdev] = allGivenYE(S, B, A, Btilde, Y, E, C_rowSums, withTitle)
    % Decide whether or not printing out metric titles.
    if nargin < 8
        withTitle = 0;
    end
    
    % Return only the title information.
    if withTitle == -1
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');  
        return
    end
        
    % Setup the dissimilarity measure to use.
    clusterDissimilarity_soft = @clusterDissimilarity_symKL;
    %clusterDissimilarity_soft = @clusterDissimilarity_cos;
    %clusterDissimilarity_soft = @clusterDissimilarity_Fisher;
    
    % Measure all metrics.
    [RE, RE_std]   = compression.evaluateClusters_RE(S, Btilde, Y, E); 
    DL             = distributionLegality(A);
    MV             = marginalValidity(B, A, Btilde, C_rowSums);
    [AE1, AE2]     = compression.evaluateClusters_AE(B, A, Y, E);
    [DD, DD_std]   = diagonalDominancy(A);
    
    [NE, NE_std]   = normalizedEntropy(Btilde);
    [CP, CP_std]   = clusterSparsity(B);
    [CS, CS_std]   = clusterSpecificity(B, C_rowSums);    
    [CDh, CDh_std] = clusterDissimilarity_hard(B, 20);
    [CDs, CDs_std] = clusterDissimilarity_soft(B);
    
    % Does not measure coherence metrics.
    CC = 0;        
    CC_std = 0;
    
    [BRh, BRh_std] = basisRank_hard(B, S);
    [BRs, BRs_std] = basisRank_soft(B, S);
    [BQh, BQh_std] = basisQuality_hard(B, S);
    [BQs, BQs_std] = basisQuality_soft(B, S);
        
    % Print out the results.
    value = sprintf('%14.6f %14.4f %14.4f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14.4f %14.4f %14.2f %14.6f %14.6f', RE, DL, MV, AE1, AE2, DD, NE, CS, CDh, CDs, CC, BRh, BRs, BQh, BQs, CP);
    stdev = sprintf('%14.6f %14s %14s %14s %14s %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14.4f %14.4f %14.2f %14.6f %14.6f', RE_std, 'NA', 'NA', 'NA', 'NA', DD_std, NE_std, CS_std, CDh_std, CDs_std, CC_std, BRh_std, BRs_std, BQh_std, BQs_std, CP_std);
    if withTitle
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', value);        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', stdev);        
    end        
end


%%
% Inner: all()
%
% Remark: 
%   - This function evaluates every possible metrics.
%
function [value, stdev] = all(S, B, A, Btilde, Cbar, C_rowSums, C, D1, D2, withTitle)
    % Decide whether or not printing out metric titles.
    if nargin < 10 
        withTitle = 0;
    end
    
    % Return only the title information.
    if withTitle == -1
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');                
        return
    end
    
    % Setup the dissimilarity measure to use.
    clusterDissimilarity_soft = @clusterDissimilarity_symKL;
    %clusterDissimilarity_soft = @clusterDissimilarity_cos;
    %clusterDissimilarity_soft = @clusterDissimilarity_Fisher;
    
    % Measure all metrics.
    [RE, RE_std]   = recoveryError(Cbar, S, Btilde);    
    DL             = distributionLegality(A);
    MV             = marginalValidity(B, A, Btilde, C_rowSums);
    [AE1, AE2]     = approximationError(C, B, A);
    [DD, DD_std]   = diagonalDominancy(A);
    
    [NE, NE_std]   = normalizedEntropy(Btilde);
    [CP, CP_std]   = clusterSparsity(B);
    [CS, CS_std]   = clusterSpecificity(B, C_rowSums);    
    [CDh, CDh_std] = clusterDissimilarity_hard(B, 20);
    [CDs, CDs_std] = clusterDissimilarity_soft(B);
    [CC, CC_std]   = clusterCoherence(B, D1, D2, 20);    
    
    [BRh, BRh_std] = basisRank_hard(B, S);
    [BRs, BRs_std] = basisRank_soft(B, S);
    [BQh, BQh_std] = basisQuality_hard(B, S);
    [BQs, BQs_std] = basisQuality_soft(B, S);
        
    % Print out the results.
    value = sprintf('%14.6f %14.4f %14.4f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14.4f %14.4f %14.2f %14.6f %14.6f', RE, DL, MV, AE1, AE2, DD, NE, CS, CDh, CDs, CC, BRh, BRs, BQh, BQs, CP);
    stdev = sprintf('%14.6f  %14s %14s %14s %14s %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14.4f %14.4f %14.2f %14.6f %14.6f', RE_std, 'NA', 'NA', 'NA', 'NA', DD_std, NE_std, CS_std, CDh_std, CDs_std, CC_std, BRh_std, BRs_std, BQh_std, BQs_std, CP_std);
    if withTitle
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s \n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', value);        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s \n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', stdev);        
    end        
end


%%
% Inner: allButD()
%
% Remark: 
%   - This function evaluates every possible metrics except the coherence.
%
function [value, stdev] = allButD(S, B, A, Btilde, Cbar, C_rowSums, C, withTitle)
    % Decide whether or not printing out metric titles.
    if nargin < 8
        withTitle = 0;
    end
    
    % Return only the title information.
    if withTitle == -1
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');                
        return
    end
    
    % Setup the dissimilarity measure to use.
    clusterDissimilarity_soft = @clusterDissimilarity_symKL;
    %clusterDissimilarity_soft = @clusterDissimilarity_cos;
    %clusterDissimilarity_soft = @clusterDissimilarity_Fisher;
        
    % Measure all metrics.
    [RE, RE_std]   = recoveryError(Cbar, S, Btilde);
    DL             = distributionLegality(A);
    MV             = marginalValidity(B, A, Btilde, C_rowSums);
    [AE1, AE2]     = approximationError(C, B, A);
    [DD, DD_std]   = diagonalDominancy(A);
    
    [NE, NE_std]   = normalizedEntropy(Btilde);
    [CP, CP_std]   = clusterSparsity(B);
    [CS, CS_std]   = clusterSpecificity(B, C_rowSums);    
    [CDh, CDh_std] = clusterDissimilarity_hard(B, 20);
    [CDs, CDs_std] = clusterDissimilarity_soft(B);
    
    [BRh, BRh_std] = basisRank_hard(B, S);
    [BRs, BRs_std] = basisRank_soft(B, S);
    [BQh, BQh_std] = basisQuality_hard(B, S);
    [BQs, BQs_std] = basisQuality_soft(B, S);
        
    % Print out the results.
    value = sprintf('%14.6f %14.4f %14.4f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14s %14.6f %14.6f %14.2f %14.6f %14.6f', RE, DL, MV, AE1, AE2, DD, NE, CS, CDh, CDs, 'NA', BRh, BRs, BQh, BQs, CP);
    stdev = sprintf('%14.6f %14s %14s %14s %14s %14.6f %14.6f %14.6f %14.6f %14.6f %14s %14.4f %14.4f %14.2f %14.6f %14.6f', RE_std, 'NA', 'NA', 'NA', 'NA', DD_std, NE_std, CS_std, CDh_std, CDs_std, 'NA', BRh_std, BRs_std, BQh_std, BQs_std, CP_std);
    if withTitle
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', value);        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', stdev);                
    end        
end


%%
% Inner: allButS()
%
% Remarks:
%   - This function evaluates every possible metrics irrelevant to the basis vectors S.
%
function [value, stdev] = allButS(B, A, Btilde, C_rowSums, C, D1, D2, withTitle)
    % Decide whether or not printing out metric titles.
    if nargin < 8
        withTitle = 0;
    end
    
    % Return only the title information.
    if withTitle == -1
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity');                
        return
    end
        
    % Setup the dissimilarity measure to use.
    clusterDissimilarity_soft = @clusterDissimilarity_symKL;
    %clusterDissimilarity_soft = @clusterDissimilarity_cos;
    %clusterDissimilarity_soft = @clusterDissimilarity_Fisher;
    
    % Measure all metrices irrelevant to the basis vectors S.
    DL             = distributionLegality(A);
    MV             = marginalValidity(B, A, Btilde, C_rowSums);
    [AE1, AE2]     = approximationError(C, B, A);
    [DD, DD_std]   = diagonalDominancy(A);
    
    [NE, NE_std]   = normalizedEntropy(Btilde);    
    [CP, CP_std]   = clusterSparsity(B);
    [CS, CS_std]   = clusterSpecificity(B, C_rowSums);    
    [CDh, CDh_std] = clusterDissimilarity_hard(B, 20);
    [CDs, CDs_std] = clusterDissimilarity_soft(B);
    [CC, CC_std]   = clusterCoherence(B, D1, D2, 20);
        
    % Print out the results.
    value = sprintf('%14s %14.4f %14.4f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14s %14s %14s %14s %14.6f', 'NA', DL, MV, AE1, AE2, DD, NE, CS, CDh, CDs, CC, 'NA', 'NA', 'NA', 'NA', CP);
    stdev = sprintf('%14s %14s %14s %14s %14s %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14s %14s %14s %14s %14.6f', 'NA', 'NA', 'NA', 'NA', 'NA', DD_std, NE_std, CS_std, CDh_std, CDs_std, CC_std, 'NA', 'NA', 'NA', 'NA', CP_std);
    if withTitle
        value = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', value);        
        stdev = sprintf('%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s\n%s', 'Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity', stdev);                
    end        
end



%%
% Inner: recoveryError()
%
% Remarks: 
%  - This function evaluates the intrinsic quality of the matrix B.
%  - This function computes the mean recovery error by averaging the
%    residual from NNLS recovery for each non-basis object in terms of 
%    basis vectors.
%  - The small is better as closer to 0.
%
function [value, stdev] = recoveryError(Cbar, S, Btilde)
    % Compute the recovery errors in a batch.
    Cbar_S = Cbar(S, :);
    residuals = Btilde'*Cbar_S - Cbar;
    
    % Measure the sum of row-wise norm of residuals averaged across N objects.
    errors = sqrt(sum(abs(residuals).^2, 2));    
    value = mean(errors);    
    stdev = std(errors);
end



%%
% Inner: conditionNumber()
%
function [value] = conditionNumber(Cbar, S)
    % Compute the recovery errors in a batch.
    Cbar_S = Cbar(S, :);
    value = cond(Cbar_S);
end



%%
% Inner: distributionLegality()
% Remark: 
%  - This function evaluates the intrinsic quality of the matrix A.
%  - This function computes the sum of all entries of A.
%  - The small is better as closer to 1
%
function value = distributionLegality(A)
    % Measure the sum of all entries of B.
    value = sum(sum(A));
end


%%
% Inner: marginalValidity()
% Remark: 
%  - This function evaluates the intrinsic quality between B and A.
%  - This function computes the difference between marginals p(z) computed
%    by summing A row-wisely and marginals computed indirectly via Bayes rule.
%  - The small is better as closer to 0.
%
function value = marginalValidity(B, A, Btilde, C_rowSums)
    % Compute the row-wise marginal distribution of A.
    A_rowSums = sum(A, 2);
    
    % Compute the marginal distribution of B via Bayes rule.
    K = size(B, 2);
    bayesResults = Btilde .* (ones(K, 1)*C_rowSums') ./ B';
    p_z = nanmean(bayesResults, 2);
    value = symmetricKL(A_rowSums, p_z);
end
    

%%
% Inner: approximationError()
% Remark: 
%  - This function evaluates the overall intrinsic quality of B and A.
%  - This function computes how much the reconstructed BAB^T is close from
%    the co-occurrence matrix C entre-wisely in terms of Frobenius norm.
%  - The second output measures approximation error without diagonal entries.
%  - The small is better as closer to 0.
%
function [value1, value2] = approximationError(C, B, A)
    % Measure how much the decopmosition is close to the original
    C_prime = B*A*B';
    value1 = norm(C - C_prime, 'fro');
    diagError = norm(diag(C) - diag(C_prime), 'fro');
    value2 = sqrt(value1^2 - diagError^2);
end


%%
% Inner: diagonalDominancy()
% Remark: 
%  - This function evaluates the overall comparative quality of A.
%  - This function computes how large the diagonal elements of A are with
%    respect to non-diagonal entries.
%  - The small is better (not necessarily close to 0).
%
function [value, stdev] = diagonalDominancy(A)
    % Measure how large diagonal entries are relative to other entries.
    A_rowSums = sum(A, 2);
    dominancies = diag(A) ./ A_rowSums;
    value = mean(dominancies);    
    stdev = std(dominancies);
end


%%
% Inner: normalizedEntropty()
% Remark: 
%  - This function evaluates the extrinstic quality of Btilde.
%  - This function computes how much each word is concentrated on several
%    topics. If the clusters are not mature enough, cluster preference for
%    each word is close to the uniform distribution.
%  - The large is better (averaged across N objects).
function [value, stdev] = normalizedEntropy(Btilde)
    % Measure how far p(z|x) is concentrated on several topics relative to the uniform distribution.
    K = size(Btilde, 1);
    entropies = entropy(Btilde) / (log(K) / log(2));
    value = mean(entropies);    
    stdev = std(entropies);
end


%%
% Inner: clusterSparsity()
% Remark: 
%   - This function evaluates how sparse each cluster is.
%   - The higher is better (averaged across K clusters).
%
function [value, stdev] = clusterSparsity(B)
    N = size(B, 1);
    colNorms_l1 = sum(abs(B), 1);
    colNorms_l2 = sqrt(sum(B.^2, 1));
    sparsities = (sqrt(N) - (colNorms_l1 ./ colNorms_l2)) ./ (sqrt(N) - 1);
    value = mean(sparsities);
    stdev = std(sparsities);
end



%%
% Inner: clusterSpecificity()
% Remark: 
%  - This function evaluates the extrinsic quality of B.
%  - This function computes how far each cluster is from the corpus
%    distribution. Note that ill-conditioned clustering (e.g: not-enough 
%    clusters) often yields the clusters similar to the corpus distribution.
%  - The larger is better (averaged across K clusters)
%
function [value, stdev] = clusterSpecificity(B, C_rowSums)
    % Measure how much each cluster is far from the corpus distribution.
    divergences = divergenceKL(B, C_rowSums);
    value = mean(divergences);
    stdev = std(divergences);
end


%%
% Inner: clusterDissimilarity_hard()
% Remark: 
%  - This function evaluates the extrinsic quality of B.
%  - This function computes how different each cluster is from other
%    clusters via counting the number of unique objects that appear in the
%    target cluster, but do not appear in all other clusters in top words.
%  - The larger is better (averaged across K clusters).
%
function [value, stdev] = clusterDissimilarity_hard(B, M)
    % Sort each group by the decreasing order of contributions.
    [~, I] = sort(B, 1, 'descend');
    
    % Pick the indices of top M contributing objects.
    [N, K] = size(B);
    M = min(M, N);
    I = I(1:M, :);
        
    % Prepare variables.
    colSet = 1:K;
    dissimilarities = zeros(1, K);
    
    % Count the number of unique objects in top M contributions.
    for k = 1:int32(K)
        currentObjects = I(:, k);
        otherObjects = unique(I(:, setdiff(colSet, k)));        
        uniqueObjects = setdiff(currentObjects, otherObjects);
        dissimilarities(k) = numel(uniqueObjects);    
    end
    
    % Measure how many of top M objects do not appear in other cluster's top M objects.
    value = mean(dissimilarities);
    stdev = std(dissimilarities);
end


%%
% Inner: clusterDissimilarity_symKL()
% Remark: 
%  - This function evaluates the extrinsic quality of B.
%  - This function computes how different each cluster is from other
%    clusters via measuring symmetric KL-divergences between the target
%    cluster and all other clusters.
%  - The larger is better (averaged across K clusters).
%
function [value, stdev] = clusterDissimilarity_symKL(B)
    % Prepare variables.
    K = size(B, 2);
    colSet = 1:K;
    dissimilarities = zeros(1, K);
    
    % Evaluate the symmetric KL-divergence between one and the others.
    for k = 1:int32(K)
        currentClusters = B(:, k);
        otherClusters = B(:, setdiff(colSet, k));
        dissimilarities(k) = mean(symmetricKL(currentClusters, otherClusters));
    end    
    
    % measure how much each cluster is different from all other clusters
    value = mean(dissimilarities);
    stdev = std(dissimilarities);
end


%%
% Inner: clusterDissimilarity_cos()
% Remark: 
%  - This function evaluates the extrinsic quality of B.
%  - This function computes how different each cluster is from other
%    clusters via measuring the cosine similarities between the target
%    cluster and all other clusters.
%  - The larger is better (averaged across all pairwise comparison).
%
function [value, stdev] = clusterDissimilarity_cos(B)
    K = size(B, 2);
    BtB = B'*B;
    offDiagonals = setdiff(1:K^2, K:K:K^2);
    dissimilarities = BtB(offDiagonals);
    value = mean(dissimilarities);
    stdev = std(dissimilarities);
end


%%
% Inner: clusterDissimilarity_Fisher()
% Remark: 
%  - This function evaluates the extrinsic quality of B.
%  - This function computes how different each cluster is from other
%    clusters via measuring the Fisher distance between the target
%    cluster and all other clusters.
%  - The larger is better (averaged across all pairwise comparison)
%
function [value, stdev] = clusterDissimilarity_Fisher(B)
    K = size(B, 2);
    sqrtB = sqrt(B);
    sqrtBtB = sqrtB'*sqrtB;
    offDiagonals = setdiff(1:K^2, K:K:K^2);
    dissimilarities = acos(sqrtBtB(offDiagonals));
    value = mean(dissimilarities);
    stdev = std(dissimilarities);    
end


%%
% Inner: clusterCoherence()
% Remark: 
%  - This function evaluates the extrinsic quality of B.
%  - This function counts how many strange word pairs (occuring alone in
%    many examples, but rarely co-occuring together in training examples)
%    exist in the top words of each cluster.
%  - The larger close to 0 is better (averaged across all clusters)
%
function [value, stdev] = clusterCoherence(B, D1, D2, L)
    % Sort each group by the decreasing order of contributions.
    [~, I] = sort(B, 1, 'descend');
    
    % Prepare variables.
    K = size(B, 2);
    coherences = zeros(1, K);
    
    % Find the coherence of each cluster
    epsilon = 0.01;
    for k = 1:int32(K)
        for i = 2:int32(L)
            top_i = I(i, k);
            for j = 1:(i - 1)
                % Smoothen the numerator by adding epsilon to avoid taking the logarithm of zero.
                top_j = I(j, k); 
                coherences(k) = coherences(k) + log((D2(top_i, top_j) + epsilon) / D1(top_j));
            end
        end
    end
    value = mean(coherences);
    stdev = std(coherences);
end



%%
% Inner: basisRank_hard()
% Remark: 
%  - This function evaluates the extrinsic quality of S with respect to B.
%  - This function computes the average rank of basis object in each cluster.
%  - No specific criteria (averaged across all clusters).
%
function [value, stdev] = basisRank_hard(B, S)
    % Sort each group by the decreasing order of contributions.
    [~, I] = sort(B, 1, 'descend');
    
    % Prepare variables.
    K = size(B, 2);
    hardRanks = zeros(1, K);
    
    % Find the rank of basis in each cluster.
    for k = 1:int32(K)
        hardRanks(k) = find(I(:, k) == S(k));
    end
    
    % Measure the mean rank.
    value = mean(hardRanks);
    stdev = std(hardRanks);
end


%%
% Inner: basisRank_soft()
% Remark: 
%  - This function evaluates the extrinsic quality of S with respect to B.
%  - This function computes the average log-likelihood difference between
%    the top object and basis object in each cluster.
%  - No specific criteria (averaged across all clusters).
%
function [value, stdev] = basisRank_soft(B, S)
    % Sort each group by the decreasing order of contributions.
    [B_sorted, ~] = sort(B, 1, 'descend');
    
    % Compute the log ratio.
    softRanks = log(B_sorted(1, :)) - log(diag(B(S, :))');
    
    % Measure the mean ratio.
    value = mean(softRanks);
    stdev = std(softRanks);
end


%%
% Inner: basisQuality_hard()
% Remark: 
%  - This function evaluates the extrinsic quality of S with respect to B.
%  - This function computes the average rank of the basis object of target
%    cluster in other clusters.
%  - The higher is better (averaged across all clusters).
%
function [value, stdev] = basisQuality_hard(B, S)
    % Sort each group by the decreasing order of contributions.
    [~, I] = sort(B, 1, 'descend');
    
    % Prepare variables.
    [N, K]= size(B);
    colSet = 1:K;
    offset = N*(0:(K-2));
    hardQualities = zeros(1, K);
    
    % Find the rank of the current basis in all other clusters.    
    for k = 1:int32(K)
        otherObjects = I(:, setdiff(colSet, k));
        hardRanks = find(otherObjects == S(k))' - offset;
        hardQualities(k) = mean(hardRanks);        
    end
        
    % measure the mean quality
    value = mean(hardQualities);
    stdev = std(hardQualities);
end


%%
% Inner: basisRank_soft()
% Remark: 
%  - This function evaluates the extrinsic quality of S with respect to B.
%  - This function computes the average likelihood of the basis object in
%    all other clusters.
%  - The smaller (must be close to 0) is better (averaged across all clusters)
%
function [value, stdev] = basisQuality_soft(B, S)
    % Prepare variables.
    K = size(B, 2);
    colSet = 1:K;
    softQualities = zeros(1, K);
    
    % Find the probability of the current basis in all other clusters.
    for k = 1:int32(K)
        otherClusters = setdiff(colSet, k);
        softRanks = B(S(k), otherClusters);
        softQualities(k) = mean(softRanks);        
    end
        
    % Measure the mean quality.
    value = mean(softQualities);
    stdev = std(softQualities);
end




%%
% Helper: divergenceKL()
% In/Outs: every computation is done between column vectors
%   - p: vector / q: vector --> return positive real: D(p || q)
%   - p: vector / q: matrix --> return row vector: D(p || each column of q)
%   - p: matrix / q: vector --> return row vector: D(each column of p || q)
%   - p: matrix / q: matrix --> return row vector: D(each column of p || each column of q)
%
function values = divergenceKL(p, q)
    % Get the sizes and perform the sanity check.
    [p_rows, p_cols] = size(p);
    [q_rows, q_cols] = size(q);
    if p_rows ~= q_rows
        error('* Incomparable distributions!');
    end
    
    % Smoothen the entries of q if they are exactly zero to avoid infinity.
    q(q < eps) = eps;
    
    % Perform column-normalization.
    q_colSums = sum(q, 1);
    q = bsxfun(@ldivide, q_colSums, q);
    
    % Horizontally duplicate column vectors if the sizes do not match.
    if p_cols > q_cols 
        q = repmat(q, [1, p_cols / q_cols]);
    elseif p_cols < q_cols
        p = repmat(p, [1, q_cols / p_cols]);
    end
    
    % Compute the divergence in terms of bit unit with the log base 2
    % Note that 0*log0 = NaN whereas its limit value is 0.
    divergences = (p .* log(p./q)) / log(2);    
    divergences(isnan(divergences)) = 0;
    
    % Return the values.
    values = sum(divergences, 1);
end


%%
%
function values = divergenceKL2(P, Q)
    % Get the sizes and perform the sanity check.
    [P_rows, P_cols] = size(P);
    [Q_rows, Q_cols] = size(Q);
    if P_rows ~= Q_rows || P_cols ~= Q_cols
        error('* Incomparable distributions!');
    end
    
    % Smoothen the entries of q if they are exactly zero to avoid infinity.
    Q(Q == 0) = eps;
    
    % Perform column-normalization.
    Q = Q / sum(sum(Q));
    
    % Compute the divergence in terms of bit unit with the log base 2.
    % Note that 0*log0 = NaN whereas its limit value is 0.
    divergences = (P .* log(P./Q)) / log(2);    
    divergences(isnan(divergences)) = 0;
    
    % Return the values.
    values = sum(sum(divergences));
end


%%
% Helper: symmetricKL()
% In/Outs: every computation is done between column vectors
%   - p: vector / q: vector --> return positive real (1 vs 1)
%   - p: vector / q: matrix --> return row vector (1 vs each column)
%   - p: matrix / q: vector --> return row vector (each column vs 1)
%   - p: matrix / q: matrix --> return row vector (each column vs each column)
%
function values = symmetricKL(p, q)
    values = 0.5*(divergenceKL(p, q) + divergenceKL(q, p));
end


%%
function values = symmetricKL2(P, Q)
    values = 0.5*(divergenceKL2(P, Q) + divergenceKL2(Q, P));
end


%%
% Helper: fisherDistance()
% In/Outs: every computation is done between column vectors
%   - p: vector / q: vector --> return positive real (1 vs 1)
%   - p: vector / q: matrix --> return row vector (1 vs each column)
%   - p: matrix / q: vector --> return row vector (each column vs 1)
%   - p: matrix / q: matrix --> return row vector (each column vs each column)
%
function values = fisherDistance(p, q)
    % Get the sizes and perform the sanity check.
    [p_rows, p_cols] = size(p);
    [q_rows, q_cols] = size(q);
    if p_rows ~= q_rows
        error('* Incomparable distributions!');
    end
    
    % Horizontally duplicate column vectors if the sizes do not match.
    % Note that each entry becomes its square root first.
    if p_cols > q_cols 
        q = sqrt(repmat(q, [1, p_cols / q_cols]));
    elseif p_cols < q_cols
        p = sqrt(repmat(p, [1, q_cols / p_cols]));
    end
    
    % Compute the Fisher information metric.
    distances = sum(p.*q, 1);
    values = acos(distances);  
end


%%
% Helper: entropy()
% In/Outs:
%   - p: vector --> real value
%   - p: matrix --> return row vector (entropy of each column vector)
%
function values = entropy(p)
    % Remember the position where the entries of p is exactly zero.
    % this is because eps * log(eps/positive) -> 0 as eps -> 0.
    I = (p == 0);
    
    % Compute the entropies in terms of bit unit with the log base 2.
    % Note that 0*log0 = NaN whereas its limit value is 0.
    entropies = -(p .* log(p)) / log(2);
    entropies(I) = 0;
    values = sum(entropies, 1);
end

    


%%
% TODO:
%



