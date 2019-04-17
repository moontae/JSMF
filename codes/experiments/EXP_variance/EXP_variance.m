function diff = EXP_variance(C)
    n = size(C, 1);
    k = 20; % 20 topics.
    nDoc = 2e3; % 2000 documents.
    nLength = 400; % 400 words per doc.
    niter = 50;
    
    [S, B, A, Btilde, Cbar, C_rowSums, diagR, C, values, elapsedTime] = ...
        factorizeC(C, k);
    alpha = estimateDirichlet(A, 'gradient');
    W = sampleFromDir(alpha', nDoc); 
    H = synthesizeDataset(B, W, nLength); 
    Cs = (H*H' - diag(sum(H,2)))/(nDoc*nLength*(nLength-1)); % Sample co-occurence.
    Ct = BuildTrueC(alpha, B);
    diff = zeros(3*niter+1,2);
    diff(1,:) = [norm(Cs-Ct,'fro'), norm(Cs-Ct)];
    Ss = {};
    for i = 1:niter
        Cs = nearestPSD(Cs, k);
        diff(3*i-1, :) = [norm(Cs-Ct,'fro'), norm(Cs-Ct)];
        if i == 1 || i == 50
            Ss{end+1} = inference.findS(Cs./sum(Cs,2), k);
        end
        Cs = Cs + (1 - sum(sum(C)))/(n^2);
        diff(3*i, :) = [norm(Cs-Ct,'fro'), norm(Cs-Ct)];
        Cs = max(Cs, 0);
        diff(3*i+1, :) = [norm(Cs-Ct,'fro'), norm(Cs-Ct)];
    end
    diff = diff./[norm(Ct,'fro'), norm(Ct)];
    fprintf(['Intersection between anchors after 1 and 50 iterations:' ...
            '%d/%d.\n'], length(intersect(Ss{1}, Ss{2})), k);
    figure('outerposition',[0 0 900 900]);
    subplot(2,1,1);
    plot(diff(:,1));
    xlim([0 50]);
    title('F-norm');
    set(gca,'XTick',1:3:151,'XTickLabel',0:50);
    subplot(2,1,2);
    plot(diff(:,2));
    xlim([0 50]);
    title('2-norm');
    set(gca,'XTick',1:3:151,'XTickLabel',0:50);
end

function Ct = BuildTrueC(alpha, B)
    k = length(alpha);
    alpha0 = sum(alpha);
    At = -alpha*alpha'/(alpha0^2*(alpha0+1));
    At(1:k+1:end) = alpha.*(alpha0-alpha)/(alpha0^2*(alpha0+1));
    At = At + alpha*alpha'/(alpha0^2);  % second moment of dirichlet
    Ct = B*At*B';   % ground truth co-occurrence
end

function C = nearestPSD(C, K)
    % Find the nearest positive semidefinite matrix with the rank K.
    [V, D] = eigs(C, K, 'LA');
    C = V * diag(max(diag(D), 0)) * V';
    C = 0.5*(C + C');
end

function W = sampleFromDir(alpha, M)
    K = length(alpha);
    W = gamrnd(repmat(alpha ,M, 1), 1, M, K);
    W = W' ./ repmat(sum(W, 2),1, K)';
end

function H = synthesizeDataset(B, W, n)
    % Parses sizing information.
    topic = B*W;
    H = mnrnd(n,topic')';
end