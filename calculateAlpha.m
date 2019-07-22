function [alpha] = calculateAlpha(err)
%This method computes adaptive parameter alpha
         numerator = 1-err;
         denominator = err;
         alpha = (1/2)*log(numerator/denominator);
end