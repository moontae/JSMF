%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Sungjun Cho & Kun Dong
% Modified: April, 2019
% Examples:
%


%%
% Main: randEig_Bows()
%
% Inputs:
%   - Z: random probe vectors, usually in the form rand(n, k+p)
%   - H: n-by-m matrix / each column is a bag-of-words vector of a document
%   - K: rank of the decomposition
%   - T: the number of power iterations for refinement of range finder
%
function [V, D] = randEig_Bows(Z, H, K, T)
    % compute normalizers
    n = sum(H, 1);
    n = n.*(n - 1);

    % compute diagonal correction terms
    d = bsxfun(@rdivide, H, n);
    d = sum(d,2);

    % Normalize H
    H = bsxfun(@rdivide, H, sqrt(n));
    
    % normalize H and d to make sure C = H*H'-diag(d) sums up to 1
    e = ones(size(H, 1), 1);
    s = e'*H*(H'*e) - sum(d);
    H = H/sqrt(s);
    d = d/s;

    Afun = @(x) H*(H'*x)-bsxfun(@times, x, d);
    lmax = powerIteration(Afun, size(H, 1));
    lmin = -lmax/10;
    shift_Afun = @(x)Afun(x) - lmin*x;
    % Range finder: Get Q an approximate basis for the range space
    AZ = shift_Afun(Z);
    [Q, ~] = qr(AZ, 0);
    for t = 1:T
        [Q, ~] = qr(shift_Afun(Q), 0);
    end

    % Should have B*(Q'*Z) = Q'*CZ; choose B to minimize least sq diff
    % The stationary equations look like
    %    symm(B*M*M'-N*M') = 0
    % or
    %    B*M*M' + M*M'*B = N*M' + M*N'
    % where M = Q'*Z and N = Q'*CZ.  This is a Sylvester equation,
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
function l = powerIteration(Afun, n)
    z = randn(n, 1);
    z = z / norm(z);
    for i = 1:20
        z = Afun(z);
        z = z/norm(z);
    end
    l = z'*Afun(z);
end




%%
% TODO:
%
