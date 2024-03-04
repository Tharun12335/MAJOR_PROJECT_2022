%% preparing data
load('mydata.mat') % including 200 observers and 120 features, 4 labels

output = grp2idx(Y);
rand_num = randperm(size(X,1));
% training data set 70%, test set 30%,
X_train = X(rand_num(1:round(0.7*length(rand_num))),:);
y_train = output(rand_num(1:round(0.7*length(rand_num))),:);

X_test = X(rand_num(round(0.7*length(rand_num))+1:end),:);
y_test = output(rand_num(round(0.7*length(rand_num))+1:end),:);
%% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true)
Mdl = fitcecoc(...
    X_train, ...
    y_train, ...
    'Learners', template, ...
    'Coding', 'onevsall',...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
%% Perform cross-validation
partitionedModel = crossval(Mdl, 'KFold', 10);
% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
% Compute validation accuracy
validation_error = kfoldLoss(partitionedModel, 'LossFun', 'ClassifError'); % validation error
validationAccuracy = 1 - validation_error;
%% test model
oofLabel_n = predict(Mdl,X_test);
oofLabel_n = double(oofLabel_n); % chuyen tu categorical sang dang double
test_accuracy_for_iter = sum((oofLabel_n == y_test))/length(y_test)*100;
%% save model
saveCompactModel(Mdl,'mySVM');
%% preparing data
load('mydata.mat')
load('dataWithNoiseSNR10dBForTest.mat');
output = grp2idx(Y);
rand_num = randperm(size(X,1));
% training data set 70%, test set 30%,
X_train = X(rand_num(1:round(0.7*length(rand_num))),:);
y_train = output(rand_num(1:round(0.7*length(rand_num))),:);
X_test = X(rand_num(round(0.7*length(rand_num))+1:end),:);
y_test = output(rand_num(round(0.7*length(rand_num))+1:end),:);
%% load and test SVM model with noise
CompactMdl = loadCompactModel('mySVM'); 
oofLabel_n = predict(CompactMdl,X_test); 
test_accuracy_for_iter = sum((oofLabel_n == y_test))/length(y_test)*100; % tinh accuracy rate
%% plotconfusion
isLabels = unique(output); 
nLabels = numel(isLabels); 
[n,p] = size(X_test);
% Convert the integer label vector to a class-identifier matrix.
[~,grpOOF] = ismember(oofLabel_n,isLabels); 
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(y_test,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [(isLabels); {''}]; 
h.YTickLabel = [(isLabels); {''}];
title('Add white Gaussian noise to original data (SNR=10dB) ','FontWeight','bold','FontSize',12);