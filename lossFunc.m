function [result] = lossFunc(h,y,weights)
%This function computes weighted training error
         tempResult = 0;
         for n=1:length(y)
                 tempResult = tempResult + weights(n).*y(n).*h(n);
         end
         result = 0.5 - (1/2)*tempResult;
end