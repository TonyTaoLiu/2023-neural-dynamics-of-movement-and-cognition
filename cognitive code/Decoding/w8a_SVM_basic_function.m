function [hitrate, C] = w8a_SVM_basic_function(Input_matrix, Labels, kfolds, ldaflag, Normalize)
% This script was written by SdT to run SVM on the w8a dataset using the LIBSVM library

Input_matrix = double(Input_matrix); %Augment data to double precision
Labels = double(Labels);

uniqueLabels = unique(Labels); %IDentify unique labels (useful when not numbers)
NumOfClasses = length(uniqueLabels); % Total number of classes
numericLabels = 1:NumOfClasses; %Numeric name of labels

if ~exist('ldaflag', 'var') %Default option is not to use lda. lda helps with lazy neurons, but overall leads to lower accuracy
    ldaflag = 0;
end

if ~exist('Normalize', 'var') %Default option is not to normalize
    Normalize = 0;
end

if Normalize == 1
    %Z-score data to normalize it. Each neuron FR is normalized to its mean across trials FR
    Input_matrix = zscore(Input_matrix);
    
end

% Transform labels into numbers, if not already 
labels_temp = Labels;
for i=1:NumOfClasses,
    idx = Labels == uniqueLabels(i);
    labels_temp(idx) = numericLabels(i);
end
Labels = labels_temp;

%Balance number of trials per class
num_trials = hist(Labels,numericLabels); %number of trials in each class
minNumTrials = min(num_trials); %find the minimum one
chosen_trials = [];
for i = 1:NumOfClasses %for each class
    idx = find(Labels == numericLabels(i)); %find indexes of trials belonging to this class
    rand_i = randsample(length(idx), minNumTrials); %Select a random n number of them
    chosen_trials = [chosen_trials; idx(rand_i)]; %Put the selected trials in a matrix, ordered by class
end
Input_matrix = Input_matrix(chosen_trials, :);
Labels = Labels(chosen_trials, :);

total_number_trials = size(Input_matrix,1); %Find the total number of trials for which a prediction needs to be made.
Predicted_labels = zeros(size(Labels)); %Initate matrix of predicted labels for each trial
cumError = 0; %Count the number of prediction errors

%Generate k-fold training/testing indices:
indices = crossvalind('Kfold', Labels, kfolds);
if kfolds == length(Labels) %If kfold equals the number of trials as for a leave-one out cross validation
    indices = 1:length(Labels);
end

% Run through each fold:
for fold = 1:kfolds
    
    testlbls = Labels(indices == fold); % test on the selected subsample 
    testdata = Input_matrix(indices == fold,:); %these are the data of the selected observations for testing
    trainlbls = Labels(indices ~= fold); %these are the labels of the selected observations for training (all folds except one)
    traindata = Input_matrix(indices ~= fold,:); %these are the data of the selected observations for training
    
    if ldaflag == 1 %If you wish to perform LDA
        ldaopts.FisherFace = 1;
        [fdaout, Weights] = fda(trainlbls, ldaopts, traindata);
        testdata = testdata*Weights; % Project testdata to same LDA subspace as traindata:
        traindata = fdaout;
    end
        
    % Train/test SVM model:
    model = svmtrain(trainlbls, traindata, '-t, 2, -q'); %train the model using a linear kernel (-t: 0) or a RBF kernel (-t: 2) and default parameters
    [svmlbls] = svmpredict(testlbls, testdata, model, '-q'); %get predicted labels given model
    
    nErr= length(find( (testlbls - svmlbls) ~= 0 )); %Find misclassifications
    cumError = cumError + nErr; %Count number of errors
    Predicted_labels(indices == fold) = svmlbls; %Keep track of the predicted labels
    
end

%Compute performance of decoder
hitrate = 1 - (cumError/total_number_trials);
%Obtain confusion matrix of predicted against real values.
C = confusionmat(Labels, Predicted_labels);
C = C ./ repmat(sum(C,2), 1, size(C,2));

end
