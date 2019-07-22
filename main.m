%external reference used
% https://www.coursera.org/learn/machine-learning/home/week/3
% https://www.coursera.org/learn/machine-learning/programming/ixFof/logistic-regression
% http://ciml.info/dl/v0_9/ciml-v0_9-ch11.pdf
% www.biostat.wisc.edu/~dpage/cs760/
% http://people.csail.mit.edu/dsontag/courses/ml12/slides/lecture13.pdf

%Main entry of the program
function   main()

% Fetching  data from test and train files

data = load('train.txt');
testData = load('test.txt');
X = data(:, 1:4);
X_test = testData(:,1:4);
y_test = testData(:,5);
y = data(:, 5);
m = length(X);

%end loading  data

%preprocessing data

% Preparing data matrix using Training dataset
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Add intercept term to test set X_test
X_test = [ones(length(X_test), 1) X_test];

%setting learning rate and no of iterations for individual logistic
%regression classifier
alpha = 0.01;
noofiterations = 1200;
initial_theta = randi([0, 1],n+1,1);

weights = ones(m,1);

%performing classification using single logistic regression

[theta_single_classifier] = gradientDescent(X, y, initial_theta, alpha, noofiterations);
result_test = 0;
accuracy_test = 0;
error_count = 0;

%  Predict output label for a test data set using single logistic
%  regression classifier
for k = 1:length(X_test)
    prob = sigmoid(X_test(k,:) * theta_single_classifier);
    if prob >=0.5
        result_test = 1;
    else
        result_test = 0;
    end
    if result_test == y_test(k)
        accuracy_test = accuracy_test + 1;
    end
    fprintf('predicted class label is %d and true class label was %d\n',result_test,y_test(k));
end

%Printing accuracy and error rate for single logistic regression classifier
error_count = length(X_test)- accuracy_test;
fprintf('Accuracy and error rate for single logistic regression classifier on test data %d%%,%d%%\n',round((error_count/length(y_test))*100),round((accuracy_test/length(y_test))*100));

%end of single logistic regression

order = input("Enter number of iterations for AdaBoost routine.Please enter value from this set [10,25,50]\n"); 

while order == 10 || order == 50 || order == 25
    theta_values = zeros(length(initial_theta),order);
    error = zeros(order, 1);
    alphas = zeros(order,1);

    %Adding initial weights to 1/size(training_data_set) corresponding to
    %each data point

    for k = 1:m
       weights(k,1) = (1/m).*weights(k,1);
    end

    
    for j = 1:order
        error(j) = 0;
        %learning parameters values for a single weak classifier using logistic regression
        initial_theta =  randi([0, 1],n+1,1);
        [theta] = gradientDescent(X, y, initial_theta, alpha, noofiterations);
        theta_values(:,j) = theta;
        y_pred_values = zeros(m,1);
        h = zeros(m,1);
   
        %calculating the output label and hypothesis values from each
        %individual classifiers learned from theta parameters
        for a=1:m
             y_pred_values(a) = predict(sigmoid(X(a,:) * theta_values(:,j)));
             h(a) =  sigmoid(X(a,:) * theta_values(:,j));
        end
   
        %calculating error
        error(j) = lossFunc(h,y,weights);
   
        % if error computed is betetr than chance(equals to 0.5) then continue
        if error(j) >= 0.5
           continue
        end
   
        %calculating adaptive parameter alpha
        alphas(j) = calculateAlpha(error(j));

        %Reweight training dataset points and normalize
        weights = updateWeights(alphas(j),y,weights,X,theta_values(:,j));
    end


    %counting no of correct labels
    correct_count = 0;

    %Predicting output label for  training data set 
    for k = 1:length(X_test)
        %getting final output label from voting by the weighted voted classifier
        finalY_Predict = returnWeightedVotedClassifer(theta_values,alphas,X_test(k,:));
   
        if finalY_Predict == 1
          y_test_result = 1;
        else
          y_test_result = 0;
        end 
    
        if(y_test_result == y_test(k))
          correct_count = correct_count + 1;
        end
   
        fprintf('For a test data  predicted class label is %d and true class label was %d\n',finalY_Predict,y_test(k));
    end

    %printing accuracy and error rate 
     p = length(X_test);
     result = correct_count/p;
     error_rate = (p - correct_count)/p;
   
     fprintf('Accuracy and Error rate for ensemble classifier after iteration %d is %d%%,%d%%\n',order,round(result*100),round(error_rate*100));
     response = input('Please enter your response to run this programme again.For Yes enter 1 and for No enter 0 ?');
   
     if response == 1
         order = input("Enter number of iterations for AdaBoost routine.Please enter value from this set [10,25,50]\n");
         continue
     else
         break
     end
   
end
          
end



