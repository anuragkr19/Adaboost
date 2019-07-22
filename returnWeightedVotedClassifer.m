function [y_Predict_Final] = returnWeightedVotedClassifer(theta_values,alphas,X_test)
%This function returns the output label of weighted voted classifier
         tempSum = 0;
         for i = 1:length(alphas) 
             tempSum = tempSum + alphas(i)*predict(sigmoid(X_test*theta_values(:,i)));
         end
         
         y_Predict_Final = sign(tempSum);
end