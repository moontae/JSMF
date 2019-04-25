function [value, stdev] = recoveryError_lr(Y, E, S, Btilde)
    if nargin == 4
        [Cbarfun, CSbar] = recoveryError_Y(Y, E, S);
    elseif nargin == 3
        Btilde = S;
        [Cbarfun, CSbar] = recoveryError_H(Y, E);
    end
    n = size(Y, 1);
    [Q, R] = qr(CSbar, 0);
    [P, ~] = qr(randn(n, 100), 0);
    CQ = Cbarfun(Q);
    res = Cbarfun(P) - CQ * (Q' * P);
    error1 = sum(res.^2, 2) * (n / 100);
    error2 = sum((CQ - (R * Btilde)').^2, 2);
    error = sqrt(error1 + error2);
    value = mean(error);
    stdev = std(error);
end

function [Cbarfun, CSbar] = recoveryError_Y(Y, E, S)
    C_rowsums = Y*sum(Y)' + sum(E, 2);
    Ybar = bsxfun(@rdivide, Y, C_rowsums);
    Ebar = bsxfun(@rdivide, E, C_rowsums);
    Cbarfun = @(x) Ybar * (Y' * x) + Ebar * x;
    CSbar = Y * Ybar(S, :)'+Ebar(:, S);
end

function [Cbarfun, CSbar] = recoveryError_H(H, S)
    m = size(H, 2);
    f = sum(H);
    f = f .* (f-1);
    Hn = bsxfun(@rdivide, H, sqrt(f));
    dn = sum(bsxfun(@rdivide, H, f), 2);
    C_rowsums = (Hn * sum(Hn)' - dn) / m;
    Hnbar = bsxfun(@rdivide, Hn, C_rowsums);
    dnbar = dn ./ C_rowsums;
    Cbarfun = @(x) (Hnbar * (Hn' * x) - dnbar .* x) / m;
    CSbar = Hn*Hnbar(S, :)';
    CSbar(S, :) = CSbar(S, :)- diag(dnbar(S));
    CSbar = CSbar / m;
end