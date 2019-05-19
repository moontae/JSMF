%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Examples:
%


%%
% Wrapper: rectifyVD()
%
% Inputs:
%   - V: N-by-K matrix
%   - D: K-by-K matrix (V and D together approximate co-occurrence C = V*D*V')
%   - K: the number of basis vectors
%   + rectifier: choose a method for jointly running rectification + compression
%     - ENN: Epsilon-NN using V*sqrt(D) as user-specified initialization
%
% Outputs:
%   - Y: NxK rectified + compressed co-occurrence
%   - E: sparse correction for ENN / counterpart for PALMs
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%
function [Y, E, elapsedTime] = rectifyVD(V, D, K, rectifier)
    switch(rectifier)      
      case 'ENN'
        % For Epsilon-NN method, E means a sparse correction.   
        [Y, E, elapsedTime] = compression.rectify_ENN([], K, 'otherwise', 150, V, D);      
            
      otherwise
        % No rectification.
        Y = V*sqrt(D);
        E = [];
        elapsedTime = 0;
    end
end




%%
% TODO:
%



