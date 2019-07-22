function [yPredict] = predict(prob)
     if prob >=0.5
        yPredict = 1;
     else
        yPredict = 0;
     end
     
end