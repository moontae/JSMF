%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: February, 2017
% Examples:
%


%%
%
function [alpha, loss, alpha0_candidates] = estimateDirichlet(A, option)
    % Computes the row-normalization.    
    K = size(A, 1);
    A_rowSums = sum(A, 2);
    Abar = bsxfun(@rdivide, A, A_rowSums);

    % Estimates moment-matching alpha0 per column individually.
    diagonals = diag(Abar)';
    colAveragesWithoutDiagonals = sum(Abar - diag(diagonals), 1) / (K-1);
    alpha0_candidates = (1 ./ (diagonals - colAveragesWithoutDiagonals)) - 1;
        
    % Defines the loss function.
    removeDiag = @(X) X - diag(diag(X));
    J_diag = @(a0) sumsqr((a0*A_rowSums + 1)/(a0 + 1) - diag(Abar));
    J_offdiag = @(a0) sumsqr(removeDiag(repmat((a0*A_rowSums')/(a0 + 1), K, 1) - Abar));
    J = @(a0) 0.5*J_diag(a0) + 0.5*J_offdiag(a0);       
    
    % Estimates based on the option.
    loss = -1;
    switch(option)
      case 'gradient'    
        % Picks the mean as the initiali alpha0 and computes the loss.
        alpha0 = mean(alpha0_candidates);        
        prevLoss = J(alpha0);        
          
        % Starts the non-linear least-square.
        eta = 1;
        gradSquareSum = 0;
        for t = 1:int32(100)            
            % Computes the gradient at the current alpha0 and reports.
            grad_J = ((1 - alpha0 - K) + alpha0*K*sumsqr(A_rowSums) + (alpha0 + 1)*sum(diag(Abar)) - (alpha0 + 1)*sum(Abar*A_rowSums))/((alpha0 + 1)^3);                                    
            %fprintf(1, '%2d: alpha0 = %.4f / grad = %.4f / moment-matching loss = %.4f\n', t, alpha0, grad_J, prevLoss);
            
            % Performs one step of gradient update.
            gradSquareSum = gradSquareSum + grad_J^2;
            alpha0 = alpha0 - (eta/sqrt(gradSquareSum))*grad_J;               
            
            % Checks the termination condition.
            newLoss = J(alpha0);                               
            if abs(newLoss - prevLoss) < 1e-9
                fprintf(1, '--> alpha0 = %.4f / moment-matching loss = %.4f\n', alpha0, newLoss);                
                break
            else
                prevLoss = newLoss;
            end
        end
        loss = newLoss;
        
      case 'baseline'
        % From Going Beyond SVD paper.        
        [~, l] = min(A_rowSums);
        u = A(l, l);
        v = A_rowSums(l);
        alpha0 = (1 - u/v) / (u/v - v);        
        loss = J(alpha0);
        fprintf(1, '--> alpha0 = %.4f / baseline loss = %.4f\n', alpha0, loss);  
    
      case 'line-search'
        % Starts line-searching.
        alpha0s = linspace(min(alpha0_candidates), max(alpha0_candidates), 1000);
        Js = zeros(length(alpha0s), 1);
        for i = 1:length(alpha0s)
            Js(i) = J(alpha0s(i));
        end    
        [minJ, minIndex] = min(Js);
        alpha0 = alpha0s(minIndex);
        fprintf(1, '--> alpha0 = %.4f / line-search loss = %.4f\n', alpha0, minJ);  
        loss = minJ;
    end

    % Returns the estimated dirichlet parameter.
    alpha = alpha0*A_rowSums;
end



