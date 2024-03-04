function class=svm(C)
load Trainfeature;
load Truelabel;
figure; title('Train');
svmStruct = svmtrain(round(Trainfeature),Truelabel,'showplot',true);
class=svmclassify(svmStruct,C,'showplot',true);
hold on;
plot(C(:,1),C(:,2),'ro','MarkerSize',12);
title('Train');
load fisheriris
X = meas;
Y = species;
cp = classperf(Y);
for i = 1:10
    [train,test] = crossvalind('LeaveMOut',Y,5);
    mdl = fitcknn(X(train,:),Y(train),'NumNeighbors',3);
    predictions = predict(mdl,X(test,:));
    classperf(cp,predictions,test);
end
disp('accuracy=');
disp((1-cp.ErrorRate)*100);
disp('Sensitivity');
disp(100-cp.Sensitivity/(38*7));
disp('specificity');
disp(100-(cp.Sensitivity/(38*7)*100));

hold off