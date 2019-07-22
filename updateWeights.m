function [weights] = updateWeights(alpha,y,weights,X,theta_values)
% This method reweight training dataset points and normalize
          Z = sum(weights);
          for i = 1:length(X)
              h = sigmoid(X(i,:)*theta_values);
              weights(i) = (1/Z).*(weights(i).*exp(-y(i)*alpha*h));
          end
end