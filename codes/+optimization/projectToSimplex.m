%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%


%%
% Main: projectToSimplex()
%
function x = projectToSimplex(y)
    u = sort(y, 'descend');
    one_minus_cumsum_u = 1 - cumsum(u);
    
    J = reshape(1:length(y), size(y));
    candidates = u + (1./J).*(one_minus_cumsum_u);
    rho = find(candidates > 0, 1, 'last');
   
    lambda = (1/rho)*one_minus_cumsum_u(rho);
    x = max(y + lambda, 0);    
end


%%
% Main: projectToSimplex()
%
% function X = projectToSimplex(Y)
%     Y = Y';
%     [N, D] = size(Y);
%     X = sort(Y, 2, 'descend');
%     X_tmp = (cumsum(X, 2)-1)*diag(sparse(1./(1:D)));
%     X = max(bsxfun(@minus, Y, X_tmp(sub2ind([N, D],(1:N)',sum(X > X_tmp,2)))),0);
%     X = X';
% end
