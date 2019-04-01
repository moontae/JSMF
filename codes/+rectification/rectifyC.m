%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
% Modified: April, 2019
% Examples:
%


%%
% Wrapper: rectifyC()
%
% Inputs:
%   - C: NxN co-occurrence matrix (joint-stochastic)
%   - K: the number of basis vectors (the number of topics)
%   - T: the number of maximum iterations (default = 15)
%
% Outputs:
%   - C:      NxN co-occurrence matrix (joint-stochastic & doubly-nonnegative)
%   + values: 2xT statistics
%     - 1st row: Changes between before and after iteration in terms of Frobenius norm
%     - 2nd row: Average square difference betweeo before and after projections in terms of Frobenius norm 
%   - elapsedTime: Total elapsed amount of seconds
%
% Remarks: 
%
function [C, values, elapsedTime] = rectifyC(C, K, rectifier)    
    switch(rectifier)
      case 'AP'
        % By the Alternating Projection.              
        [C, values, elapsedTime] = rectification.rectifyC_AP(C, K);
        
      case 'DC'
        % By the Diagonal Completion.        
        [C, values, elapsedTime] = rectification.rectifyC_DC(C, K);        
        
      case 'DP'
        % By the Dykstra Projection. 
        [C, values, elapsedTime] = rectification.rectifyC_DP(C, K);
              
      otherwise
        % No rectification.  
        values = [];
        elapsedTime = 0;
     end
end




%%
% TODO:
%


