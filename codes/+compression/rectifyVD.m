%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%


%%
% Wrapper: rectifyVD()
%
% Inputs:
%   - V: 
%   - D: 
%   - K: 
%   - rectifier:
%
% Outputs:
%   - Y:
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%
function [Y, E, elapsedTime] = rectifyVD(V, D, K, rectifier)
    switch(rectifier)      
      case 'ENN'
        % For Epsilon-NN method, E means a sparse correction.   
        [Y, E, elapsedTime] = compression.rectify_ENN([], K, 'otherwise', 50, V, D);      
            
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



