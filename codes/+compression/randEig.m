%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Kun Dong
% Examples:
%


%%
% Main: randEig()
%
% Inputs:
%   - Z: random probe vectors, usually in the form rand(N, 2*K)
%   - H: NxM matrix where each column is a bag-of-words vector for an example
%   - K: rank of the decomposition
%   - T: the number of power iterations for refinement of range finder
%
% Outputs:
%   - V: NxN matrix
%   - D: KxK matrix
%
function [V, D] = randEig(Z, A, K, T)
    lmax = powerIteration(A);
    lmin = -lmax/10;
    shift_A = @(x)A*x - lmin*x;
    if nargin < 4
        T = 0; 
    end
    
    % Range finder: Get Q an approximate basis for the range space
    AZ = shift_A(Z);
    [Q, ~] = qr(AZ, 0);
    for t = 1:T
        [Q, ~] = qr(shift_A(Q), 0);
    end
    % Should have B*(Q'*Z) = Q'*AZ; choose B to minimize least sq diff
    % The stationary equations look like
    %    symm(B*M*M'-N*M') = 0
    % or
    %    B*M*M' + M*M'*B = N*M' + M*N'
    % where M = Q'*Z and N = Q'*AZ.  This is a Sylvester equation,
    % which we solve via a Bartels-Stewart approach.

    M = Q'*Z;
    N = Q'*AZ;
    C = N*M';
    C = C + C';
  
    [~, S, V] = svd(M', 0);
    s2 = diag(S).^2;
    e = ones(size(s2));
    Bt = (V'*C*V) ./ (s2 * e' + e * s2');
    B = V*Bt*V';

    % Reconstruct the eigenpairs
    [V, D] = eig(B);
    [~, I] = sort(diag(D), 'descend');
    D = max(D(I(1:K), I(1:K)) + lmin, 0);
    V = Q*V(:, I(1:K));
end


%%
% Inner: powerIteration()
%
function l = powerIteration(A)
    n = length(A);
    z = randn(n, 1);
    z = z / norm(z);
    for i = 1:20
        z = A*z;
        z = z/norm(z);
    end
    l = z'*A*z;
end




%%
% TODO:
%
