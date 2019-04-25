%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: David Bindel & Kun Dong
% Examples: 
%


%%
% Main: solveSCLS_activeSet(A, b)
%
% Inputs:
%   - A: NxK matrix that has coefficients for a linear system
%   - b: Nx1 vector 
%
% Outputs:
%   - x: Kx1 vector that has the best solution satisfying Ax=b
%
% Remark:
%   - This function finds a least square x that minimize norm(A*x-b)^2/2 
%     subject to the simplex constraint x >= 0 and sum(x) = 1.
%   - At exit, it should satisfy A'*(b-A*x) + w - l = 0 roughly at machine precision.
%
function [x, w, l] = solveSCLS_activeSet(A, b)
    % Reduce to a square simplex problem.
    [m, n] = size(A);
    [Q, R] = qr(A, 0);
    b = Q'*b;

    % First k indices in permutation p are in the passive set; start empty.
    p = 1:n;
    k = 0;

    % Initial guess and dual.
    x = zeros(n, 1);
    r = R'*(b - R*x);
    w = r - x'*r;

    % Max inner iterations allowed.
    iter = 0;
    itmax = 30*n;

    % Tolerance for step zero convergence.
    normR1 = norm(R, 1);
    normRinf = norm(R, inf);
    tol = 2*n*eps*normRinf*norm(b, inf);

    % Outer loop: add free variables.
    while k < n && any(w(k+1:n) > tol)

        % Move index with largest dual into the passive set.
        [wt, t] = max(w(k+1:n));
        [p, x, w, R, b] = givUpdate(k + t, k + 1, p, x, w, R, b);
        k = k+1;

        % Figure out where we would like to go next.
        c = R(1:k, 1:k)' \ ones(k, 1);
        l = (1 - c'*b(1:k)) / (c'*c);
        s = R(1:k, 1:k) \ (b(1:k) + l*c);

        % Inner loop to add constraints.
        while any(s <= 0) && iter < itmax
             iter = iter + 1;

             % Find step size and the constraint to activate.
             QQ = find(s <= 0);
             if any(x(QQ) <= 0)
                 alpha = 0;
                 [~, ts] = min(x(QQ));
                 t = ts(1);
             else
                 [alpha,t] = min(x(QQ) ./ (x(QQ)-s(QQ)));
                 t = QQ(t);
             end

             % Move to the first binding constraint (x(t) = 0).
             x(1:k) = x(1:k) + alpha*(s - x(1:k));
             x(t) = 0;

             % Move index t into the active set.
             [p, x, w, R, b] = givDowndate(t, k, p, x, w, R, b);
             k = k - 1;

             % Recompute s with new constraint set.
             c = R(1:k, 1:k)' \ ones(k, 1);
             l = (1 - c'*b(1:k)) / (c'*c);
             s = R(1:k, 1:k) \ (b(1:k) + l*c);
        end

        x(:) = 0;
        x(1:k) = s;
        r = R'*(b - R*x);
        w = r - x'*r;
        w(1:k) = 0;

        normR1 = norm(R, 1);
        normRinf = norm(R, inf);
        tol = 2*n*eps*normR1*(2*normRinf*norm(x, inf) + norm(b, inf));
    end

    x(p) = x;
    w(p) = w;
end


%%
% Inner: givUpdate()
%
% Remark: 
%   - Use Givens rotations to move column t up to k
%
function [p, x, w, R, b] = givUpdate(t, k, p, x, w, R, b)
    for j = t-1:-1:k
        [p, x, w, R, b] = givSwap(j, p, x, w, R, b);
    end
end


%%
% Inner: giveDowndate()
%
% Remark:
%   - Use Givens rotations to move column t down to k
%
function [p, x, w, R, b] = givDowndate(t, k, p, x, w, R, b)
    for j = t:k-1
        [p, x, w, R, b] = givSwap(j, p, x, w, R, b);
    end
end


%%
% Inner: givSwap()
%
% Remark:
%   - Use Givens rotations to swap vars k and k+1
%
function [p, x, w, R, b] = givSwap(k, p, x, w, R, b)
    p([k, k+1]) = p([k+1, k]);
    x([k, k+1]) = x([k+1, k]);
    w([k, k+1]) = w([k+1, k]);
    R(:, [k, k+1]) = R(:, [k+1, k]);
    G = givens(R(k, k), R(k+1, k));
    R([k, k+1], k:end) = G*R([k, k+1],k:end);
    b([k, k+1]) = G*b([k, k+1]);
    R(k+1, k) = 0;
end




%%
% TODO:
%