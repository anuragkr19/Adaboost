function [maxValue] = predictMax(y_pred_values,X_test,theta_values,m)

    countOne = 0;
    countZero = 0;
    
    for p = 1:m
        %theta_values(:,p)
        y_pred = predict(sigmoid(X_test * theta_values(:,p)));
        if y_pred == 1
            countOne = countOne + 1;
        else
            countZero = countZero + 1;
        end            
    end
    
    if countOne > countZero
        maxValue = 1;
    else
        maxValue = 0;
    end
end