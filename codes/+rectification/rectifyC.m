%% 
% Joint Stochastic Matrix Factorization (JSMF)
%
% Coded by: Moontae Lee
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
%   - C: NxN co-occurrence matrix (joint-stochastic & doubly-nonnegative)
%   + values: 2xT statistics
%     - 1st row: changes between before and after iteration in terms of Frobenius norm
%     - 2nd row: average square difference betweeo before and after projections in terms of Frobenius norm 
%   - elapsedTime: total elapsed amount of seconds
%
% Remarks: 
%   - This function wraps multiple different algorithms for rectification.
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
              
      case 'DR'
        % By the Douglas-Rachford Projection.
        [C, values, elapsedTime] = rectification.rectifyC_DR(C, K);
        
      otherwise
        % No rectification.  
        values = [];
        elapsedTime = 0;
     end
end




%%
% TODO:
%


