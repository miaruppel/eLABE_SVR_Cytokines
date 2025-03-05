function [ results_struct ] = svm_scripts_v2021_matlab( rmat, labels, numFeatures, testRmat, testLabels, option)

%Last Updated: 01.22.21 Ashley Nielsen
%
% 02.12.19 Ashley Nielsen
%   Added accompanying tenfoldCV_wrapper.m last located at
%   /data/cn5/ashley/tenfoldCV_wrapper.m
%
% 05.08.17 Ashley Nielsen
%   Added multi-class SVM
%   Added a test individuals on multi-class SVM
%
% 03.28.17 Ashley Nielsen
%   Added an option to not do LOOCV (useful for train/test sets)
%
% 08.18.16 Ashley Nielsen
%   Added the ability to not do feature selection
%   Made output a structure
%
% 07.22.16 Ashley Nielsen
%   Fixed issue if the input isn't square in variablefeatureselector
%
% 06.14.16 Ashley Nielsen
%   Fixed up the R to Z norm so that don't do this if input > 1
%   Made outputs struct instead of many variables
%   Added a re-calculation of SVM/SVR using only consensus features
%
% 03.02.16 Ashley Nielsen
%   Cleaned up the outputs so they are more useful

% This script can run SVM or SVR depending on the inputs. This script can
% also test a new dataset on models trained from another dataset.

% INPUTS:
%   rmat - numROIs x numROIs x numSubjects matrix of correlations
%           *In the normal SVM/SVR condition, these are the data you wish to
%           classify/predict label
%           *In the test SVM/SVR condition, these are the data you are
%           using to classify/predict the label of your test data

%   labels - numSubjects list of data
%           *In the SVM condition, this should be list of 1s and -1s which
%           corresponds to the groups each subject in rmat belong to
%           *In the SVR condition, this should be the variables that you
%           want to predict (ex: age, meanFD, tic score...)
%           *In the multi-class SVM condition, this should be a
%           numSubjects X numClasses matrix with ones and -ones that looks
%           like this:
%                   Class 1 Class 2 Class 3 Class 4
%                       1       -1     -1      -1      <=Class 1 label
%                      -1       -1      1      -1      <=Class 3 label

%   numFeatures - list of the number of features to test ex: [200, 400, 600]
%           *If you would like to do NO feature selection, make this 0

%   testRmat - 264 x 264 x number of test subjects matrix of correlations
%           *If just typical SVM/SVR make this 0
%           *Haven't figured out what I'm going to do for the SVM condition
%           *In the SVR condition, these are the data you want to test on
%           the models from rmat (Ex: testing TS kids on a developmental
%           dataset predicting age)

%   testLabels - number of test subjects list of labels
%           *If just typical SVM/SVR make this 0
%           *In the SVM condition, theare are the labels for the test
%           subjects (Ex: 1s if all novel TS kids you are testing against a
%           TS/NC classification model)
%           *In the SVR condition, these are the labels for the test
%           subjects (Ex: ages of TS kids you are testing against a
%           developmental dataset)

%   option - 0 for most cases
%            1 if data for SVM is paired
%               *If paired, rmat should be ordered all group1 subjects first
%               and group2 subjects second, and pairs should have the same
%               numerical ordering
%            2 if you do not want to do LOOCV
%               *Only valid if you have a separate training and testing set
%               for either SVM/SVR
%            3 if you want to do random feature selection

% OUTPUTS:
%   predictedLabels - numSubjects x length(numFeatures) matrix of predicted
%           class/continuous labels
%           *In the SVM condition, this will be a list of 1s and 0s which
%           will tell you whether that subject was predicted correctly (1 =
%           correct, 0 = incorrect) at each numFeatures
%           *In the SVR condition, this will be a list of the SVR predicted
%           labels for each subject at each numFeatures (ex: predicted age)

%   featureList - max(numFeatures) x 2 x numSubjects x length(numFeatures)
%           list of the ROI pairs determined for the SVM/SVR prediction

%   featureWeights - max(numFeatures) x numSubjects x length(numFeatures)
%           weights of features used by the SVM and SVR algorithms

%   performanceMeasure - various measures of performance
%           *In the SVM condition, this will give you the hitRate (%
%           correct) for each numFeatures tested
%           *In the SVR condition, this will give you the R^2 or the
%           variance explained by the these features
%           *In the SVM test subjects condition, this will do something...
%           *In the SVR test subjects condition, this will give you the
%           predicted labels for the test subjects

% USAGE:
%       For SVM - result_struct = svm_scripts(corrmat,classLabels,[200:200:1000],0,0,0);
%
%       For SVR - result_struct = svm_scripts(corrmat,ages,[200:200:1000],0,0,0);
%
%       For testing subjects with SVR - result_struct = 
%       svm_scripts(typ_dev_corrmat,typ_dev_ages,[200:200:1000],ts_corrmat,ts_ages,0);
%
%       NOTE: If you want to do L20CV, I recommend looping your data
%               EX-
%                   for i = 1:N-1
%                       for j = i+1:N
%                           test_idx = [i j]; % 2 left out
%                           train_idx = setdiff(1:N,test_idx);
%                           results{i,j} =
%                           svm_scripts(corrmat(:,:,train_idx),labels(train_idx),
%                           [200:200:1000],corrmat(:,:,test_idx),labels(test_idx),2);
%                           results{j,i} = results{i,j}; % test 2 left out
%                           using a single model
%                       end
%                   end
%
%      NOTE: If you want to do 10-fold cross validation--
%
%           test_idx = round(linspace(1,length(labels),11));
%           for i = 1:10
%               train_idx = setdiff(1:length(labels),test_idx(i):test_idx(i+1)-1);
%               results{i} =
%               svm_scripts_beta(corrmat(:,:,train_idx),labels(train_idx),0,
%               corrmat(:,:,test_idx(i):test_idx(i+1)-1),labels(test_idx(i+1)-1),2);
%           end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Determine what analysis is being done
% Cases:
%   1 - SVM
%   2 - SVR
%   3 - SVM - test subs
%   4 - SVR - test subs
%   5 - SVM - paired
%   6 - SVM - multi class
%   7 - SVM - multi class - test subs



%% Initialize Variables
consensusFeatureMat = zeros(size(rmat,1),size(rmat,2),length(numFeatures));


%Check if SVR
LNOCV = 0;
svr_check = false;
svm_multi_check = false;
if numel(unique(labels)) > 2 %If more than 2 classes, SVR
    if length(unique(labels)) < 5
        svm_multi_check = true;
    else
    % SVR
    svr_check = true;
    end
end


[numRows numCols numSubjects] = size(rmat);
noFeatSelect = false;
% Check if NO feature selection
if numFeatures == 0
    % Need to find out how many features are actually here
    noFeatSelect = true;
    if numRows == numCols %Make sure square
        center_ones = 0;
        for i = 1:numRows
            if rmat(i,i,1)~=0
                center_ones = center_ones + 1;
            end
        end
        numFeatures = length(nonzeros(triu(sum(rmat,3))))-center_ones;
        consensusFeatureMat = sum(rmat,3)~=0;
    else
        numFeatures = length(nonzeros(sum(rmat,3)));
    end
    %numFeatures = (numRows*numRows-numRows)/2;
end

%Check if testing subs
test_check = false;
if length(testLabels) ~= 1
    % test subs
    test_check = true;
end


if svr_check
    if test_check
        a_type = 4;
    else
        if LNOCV == 0
            a_type = 2;
        else
            a_type = 7;
        end
    end
else
    if ~svm_multi_check
        if test_check
            a_type = 3;
        else
            if option == 1
                a_type = 5;
            else
                a_type = 1;
            end
        end
    else
        
        if test_check
            a_type = 7;
        else
            a_type = 6;
        end
        
    end
end

%% Initialize Variables
numSubjects = length(labels);
predictedLabels = zeros(length(labels),length(numFeatures));
featureList = zeros(max(numFeatures),2,numSubjects,length(numFeatures));
featureWeights = zeros(max(numFeatures),numSubjects,length(numFeatures));



%Re-organize labels
if size(labels,1) == 1
    labels = labels';
end

%Convert R Correlations to Z values (Normalization)

if max(max(max(abs(rmat)))) <= 1 %WTF why is 1 not equal to 1
    if length(unique(unique(unique(rmat))))~=2
    rmat = convertR2Z(rmat);
    if testRmat ~= 0
        testRmat = convertR2Z(testRmat);
    end
    else
        disp('Binary values so skipping r to z transform...')
    end
else
    disp('Values greater than 1 so skipping r to z transform...')
end
% not sure why I did this, but it sure screws up SVM train/test
% if size(rmat,1) == size(rmat,2) % If square
%     for n = 1:numSubjects
%         rmat(:,:,n) = transpose(rmat(:,:,n))+rmat(:,:,n);
%     end
% end
rmat(rmat==Inf)=1;
%% Main Sequence


switch a_type
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SVM
    case 1  
        
        if ~noFeatSelect % only have to do this if you want to do any feature selection
        
        %Loop through and Leave out One for Cross Validation
        for i = 1:numSubjects
            %Leave out one subject
            disp(['Leave out Subject # ',num2str(i)])
            if i == 1
                tempLabels = labels(2:end);
                tempRmat = rmat(:,:,2:end);
                testLabel = labels(1);
            elseif i == numSubjects
                tempLabels = labels(1:numSubjects-1);
                tempRmat = rmat(:,:,1:numSubjects-1);
                testLabel = labels(numSubjects);
            else
                tempLabels = cat(1,labels(1:i-1),labels(i+1:end));
                tempRmat = cat(3,rmat(:,:,1:i-1),rmat(:,:,i+1:end));
                testLabel = labels(i);
            end
            
            %Pick Top numFeatures from Training Data
            [featureList_temp,features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option);
            
            for f = 1:length(numFeatures)
                nonZFeatures = length(nonzeros(featureList_temp(:,1,f)));
                featureList(1:nonZFeatures,:,i,f) = featureList_temp(1:nonZFeatures,:,f);
                
                disp('Training and Testing SVM')
                if numFeatures<=length(featureList_temp)
                    trainFeatures = features(:,1:numFeatures(f),f);
                    %trainData = data(features(:,1:numFeatures(f),f),tempLabels);
                    
                    %Extract features from Testing Data
                    testFeatures = zeros(1,numFeatures(f));
                    for j = 1:numFeatures(f)
                        testFeatures(j) = rmat(featureList(j,1,i,f),featureList(j,2,i,f),i);
                    end
                else
                    trainFeatures = features(:,:,f);
                    %trainData = data(features(:,:,f),tempLabels);
                    
                    %Extract features from Testing Data
                    %testFeatures = zeros(1,numFeatures(f));
                    for j = 1:length(featureList_temp)
                        testFeatures(j) = rmat(featureList(j,1,i,f),featureList(j,2,i,f),i);
                    end
                                 
                end
                
                mdl = fitcsvm(trainFeatures,tempLabels);
                
                tst = predict(mdl,testFeatures);
                
                %testData = data(testFeatures,testLabel);
                
                %tst = test(a,testData);
                
                
                %Get featureWeights
                if numFeatures<=length(featureList_temp)
                    %featureWeights(1:numFeatures(f),i,f) = get_w(a)';
                    featureWeights(1:numFeatures(f),i,f) = mdl.Beta;
                else
                    %featureWeights(1:length(featureList_temp),i,f) = get_w(a)';
                    featureWeights(1:length(featureList_temp),i,f) = mdl.Beta;
                end
                
                if tst == testLabel
                    predictedLabels(i,f) = 1;
                else
                    predictedLabels(i,f) = 0;
                end
            end
            clear trainFeatures mdl tst tempLabels tempRmat testLabel testFeatures
        end
        
        % RESULTS
        hitRate = sum(predictedLabels)./numSubjects.*100;
        performanceMeasure = hitRate;
        for f = 1:length(numFeatures)
            results_struct(f).predictedLabels = predictedLabels(:,f);
            results_struct(f).featureList = featureList(:,:,:,f);
            results_struct(f).featureWeights = featureWeights(:,:,f);
            results_struct(f).hitRate = hitRate(f);
        end
        
        performanceName = 'hitRate';
        else % easy LOOCV because no feature selection
            
            % Linearize
            features = zeros(numFeatures,numSubjects);
            if size(rmat,1) == size(rmat,2)
                count = 1;
                for i = 1:size(rmat,1)
                    for j = i+1:size(rmat,1)
                        features(count,:) = squeeze(rmat(i,j,:));
                        count = count+1;
                    end
                end
            else
                count = 1;
                for i = 1:size(rmat,1)
                    for j = 1:size(rmat,2)
                        features(count,:) = rmat(i,j,:);
                        count = count+1;
                    end
                end
            end
         
            disp('Training....')
            mdl = fitcsvm(features',labels,'Leaveout','on');
                   
            disp('Testing....')
            tst = kfoldPredict(mdl);
            
            for i = 1:length(tst)
                if tst(i) == labels(i)
                    predictedLabels(i) = 1;
                else
                    predictedLabels(i) = 0;
                end
                
                % get feature weights
                featureWeights(:,i) = mdl.Trained{i}.Beta;
            end

            % get feature weights
            
            hitRate = sum(predictedLabels)./numSubjects.*100;
            performanceMeasure = hitRate;
            
            results_struct.predictedLabels = predictedLabels;
            results_struct.featureWeights = featureWeights(:,:);
            results_struct.hitRate = hitRate;
            
        end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% SVR
    case 2  
        if ~noFeatSelect % only have to do this if you want to do any feature selection
        
        %Loop through and Leave out One for Cross Validation
        for i = 1:numSubjects
            %Leave out one subject
            disp(['Leave out Subject # ',num2str(i)])
            if i == 1
                tempLabels = labels(2:end);
                tempRmat = rmat(:,:,2:end);
                testLabel = labels(1);
            elseif i == numSubjects
                tempLabels = labels(1:numSubjects-1);
                tempRmat = rmat(:,:,1:numSubjects-1);
                testLabel = labels(numSubjects);
            else
                tempLabels = cat(1,labels(1:i-1),labels(i+1:end));
                tempRmat = cat(3,rmat(:,:,1:i-1),rmat(:,:,i+1:end));
                testLabel = labels(i);
            end
            
            %Pick Top numFeatures from Training Data
            [featureList_temp,features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option);
            
            for f = 1:length(numFeatures)
                %Extract features from Testing Data
                nonZFeatures = length(nonzeros(featureList_temp(:,1,f)));
                testFeatures = zeros(1,nonZFeatures);
                featureList(1:nonZFeatures,:,i,f) = featureList_temp(1:nonZFeatures,:,f);
                
                for j = 1:nonZFeatures
                    testFeatures(j) = rmat(featureList(j,1,i,f),featureList(j,2,i,f),i);
                end
                %disp('Training and Testing SVR')
                %trainData = data(features(:,1:nonZFeatures,f),tempLabels);
                trainFeatures = features(:,1:nonZFeatures,f);
                
                %testData = data(testFeatures,testLabel);
                                
                mdl = fitrsvm(trainFeatures,tempLabels);
                
                tst = predict(mdl,testFeatures);
                
                %testData = data(testFeatures,testLabel);
                
                %tst = test(a,testData);
                
                
                %Get featureWeights
                if numFeatures<=length(featureList_temp)
                    %featureWeights(1:numFeatures(f),i,f) = get_w(a)';
                    featureWeights(1:numFeatures(f),i,f) = mdl.Beta;
                else
                    %featureWeights(1:length(featureList_temp),i,f) = get_w(a)';
                    featureWeights(1:length(featureList_temp),i,f) = mdl.Beta;
                end
                
                predictedLabels(i,f) = tst;
                
            end
            clear trainFeatures mdl tst tempLabels tempRmat testLabel testFeatures
            
        end
        
        % RESULTS
        performanceMeasure = corr(labels,predictedLabels).^2;
        
        for f = 1:length(numFeatures)
            results_struct(f).predictedLabels = predictedLabels(:,f);
            results_struct(f).featureList = featureList(:,:,:,f);
            results_struct(f).featureWeights = featureWeights(:,:,f);
            results_struct(f).R2 = performanceMeasure;
        end
        
        performanceName = 'R2';
        
        
        %                 %SVR
        %                 ker = kernel('linear',2);
        %                 algo = svr(ker);
        %                 algo.epsilon = 0.00001;
        %                 algo.optimizer = 'andre';
        %                 algo.C = Inf;
        %                 algo.balanced_ridge = 0;
        %
        %                 [tr a] = train(algo,trainData);
        %                 tst = test(a,testData);
        %
        %                 %Get featureWeights
        %                 featureWeights(1:numFeatures(f),i,f) = get_w(a);
        %
        %                 %tst.X
        %                 predictedLabels(i,f) = tst.X;
        %                 clear trainData tr a testData tst
        %             end
        %             clear tempLabels tempRmat testLabel
        %         end
        %         performanceMeasure = corr(labels,predictedLabels).^2;
        %         for f = 1:length(numFeatures)
        %             results_struct(f).predictedLabels = predictedLabels(:,f);
        %             results_struct(f).featureList = featureList(:,:,:,f);
        %             results_struct(f).featureWeights = featureWeights(:,:,f);
        %             results_struct(f).R2 = performanceMeasure(f);
        %         end
        %         performanceName = 'R2';
        else % You get to do the easy SVR
            % Linearize
            features = zeros(numFeatures,numSubjects);
            if size(rmat,1) == size(rmat,2)
                count = 1;
                for i = 1:size(rmat,1)
                    for j = i+1:size(rmat,1)
                        features(count,:) = squeeze(rmat(i,j,:));
                        count = count+1;
                    end
                end
            else
                count = 1;
                for i = 1:size(rmat,1)
                    for j = 1:size(rmat,2)
                        features(count,:) = rmat(i,j,:);
                        count = count+1;
                    end
                end
            end
            
            disp('Training....')
            mdl = fitrsvm(features',labels,'Leaveout','on');
            
            disp('Testing....')
            tst = kfoldPredict(mdl);
            
            for i = 1:length(tst)
                predictedLabels(i) = tst(i);
                % get feature weights
                featureWeights(:,i) = mdl.Trained{i}.Beta;
            end
                
                
            end
            
                   % RESULTS
        performanceMeasure = corr(labels,predictedLabels).^2;
        
        for f = 1:length(numFeatures)
            results_struct(f).predictedLabels = predictedLabels(:,f);
            results_struct(f).featureList = featureList(:,:,:,f);
            results_struct(f).featureWeights = featureWeights(:,:,f);
            results_struct(f).R2 = performanceMeasure;
        end
        
        performanceName = 'R2';
         
            
            
            
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SVM with test subjects
    case 3  
        %testRmat = convertR2Z(testRmat);
        
        %Re-organize labels
        if size(testLabels,1) == 1
            testLabels = testLabels';
        end
        
        if option == 2 % If you do not want to do LOOCV
            featureList = zeros(max(numFeatures),2,length(numFeatures));
            featureWeights = zeros(max(numFeatures),length(numFeatures));
            predictedLabels = zeros(length(testLabels),length(numFeatures));
            hits = zeros(length(testLabels),length(numFeatures));
            
            %Single Model in the Training Set
            disp('Create Single Model in Training Set')
            [featureList_temp,features] = variablefeatureSelector(rmat,labels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option); % select all features
            
            
            for f = 1:length(numFeatures)
                disp('Training and Testing SVM')
                nonZFeatures = length(nonzeros(featureList_temp(:,1,f)));
                featureList(1:nonZFeatures,:,f) = featureList_temp(1:nonZFeatures,:,f);
                
                trainFeatures = features(:,1:numFeatures(f),f);
                %trainData = data(features(:,1:nonZFeatures,f),labels);
                testFeatures = zeros(length(testLabels),nonZFeatures);
                for t = 1:length(testLabels)
                    for j = 1:nonZFeatures
                        testFeatures(t,j) = testRmat(featureList(j,1,f),featureList(j,2,f),t);
                    end
                end  
                
                
                mdl = fitcsvm(trainFeatures,labels);
                
                tst = predict(mdl,testFeatures);
                
                featureWeights(1:nonZFeatures,f) = mdl.Beta;
                
                predictedLabels(:,f) = (tst==testLabels);
                
                 
                    
                    clear testsFeatures testsData
                
                
            end
            hitRate = sum(predictedLabels)./length(testLabels).*100;
            performanceMeasure = hitRate;
            performanceName = 'testHitRate';
            
        else % If you want to do LOOCV and get a confidence interval for prediction
            
            predictedTestLabels = zeros(length(testLabels),numSubjects,length(numFeatures));
            
            % N Models in the Training Set
            %Loop through and Leave out One for Cross Validation
            for i = 1:numSubjects
                %Leave out one subject
                disp(['Leave out Subject # ',num2str(i)])
                if i == 1
                    tempLabels = labels(2:end);
                    tempRmat = rmat(:,:,2:end);
                    testLabel = labels(1);
                elseif i == numSubjects
                    tempLabels = labels(1:numSubjects-1);
                    tempRmat = rmat(:,:,1:numSubjects-1);
                    testLabel = labels(numSubjects);
                else
                    tempLabels = cat(1,labels(1:i-1),labels(i+1:end));
                    tempRmat = cat(3,rmat(:,:,1:i-1),rmat(:,:,i+1:end));
                    testLabel = labels(i);
                end
                
                %Pick Top numFeatures from Training Data
                [featureList,features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option);
                
                for f = 1:length(numFeatures)
                    
                    disp('Training and Testing SVM')
                    trainData = data(features(:,1:numFeatures(f),f),tempLabels);
                    
                    [tr a] = train(svm,trainData);
                    
                    %Extract features from Testing Data
                    testFeatures = zeros(1,numFeatures(f));
                    for j = 1:numFeatures(f)
                        testFeatures(j) = rmat(featureList(j,1,f),featureList(j,2,f),i);
                    end
                    testData = data(testFeatures,testLabel);
                    tst = test(a,testData);
                    
                    testsFeatures = zeros(length(testLabels),numFeatures(f));
                    for t = 1:length(testLabels) 
                        for j = 1:numFeatures(f)
                            testsFeatures(t,j) = testRmat(featureList(j,1,f),featureList(j,2,f),t);
                        end  
                    end
                    testsData = data(testsFeatures,testLabels);
                    testsTST = test(a,testsData);
                        
                    predictedTestLabels(:,i,f) = testsTST.X;
                    %Get featureWeights
                    featureWeights(1:numFeatures(f),i,f) = get_w(a);
                    
                    %tst.X
                    predictedLabels(i,f) = tst.X;
                    clear trainData tr a testData tst
                end
                clear tempLabels tempRmat testLabel
            end
            performanceMeasure = predictedTestLabels;
            performanceName = 'predictedTestLabels';
        
        % RESULTS
        hitRate = sum(predictedLabels)./numSubjects.*100;
        end
        
        
        
        performanceMeasure = hitRate;
        for f = 1:length(numFeatures)
            results_struct(f).predictedLabels = predictedLabels(:,f);
            results_struct(f).featureList = featureList(:,:,:,f);
            results_struct(f).featureWeights = featureWeights(:,:,f);
            results_struct(f).hitRate = hitRate(f);
            
        end

        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SVR with test subjects
    case 4  
        %testRmat = convertR2Z(testRmat);
        
        %Re-organize labels
        if size(testLabels,1) == 1
            testLabels = testLabels';
        end
        
        if option == 2 % If you don't want to do LOOCV with a test set
            predictedLabels = zeros(length(testLabels),length(numFeatures));
            featureWeights = zeros(max(numFeatures),length(numFeatures));
            
            %Single Model in the Training Set
            disp('Create Single Model in Training Set')
            [featureList_temp,features] = variablefeatureSelector(rmat,labels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option); % select all features
            
            for f = 1:length(numFeatures)
                %Extract features from Testing Data
                nonZFeatures = length(nonzeros(featureList_temp(:,1,f)));
                featureList(1:nonZFeatures,:,f) = featureList_temp(1:nonZFeatures,:,f);
                
                %disp('Training and Testing SVR')
                             
                %trainData = data(features(:,1:nonZFeatures,f),labels);
                trainFeatures = features(:,1:nonZFeatures,f);

                disp('Training and Testing....')
                mdl = fitrsvm(trainFeatures,labels);
                
                testsFeatures = zeros(length(testLabels),nonZFeatures);
                for t = 1:length(testLabels)
                    for j = 1:nonZFeatures
                        testsFeatures(t,j) = testRmat(featureList(j,1,f),featureList(j,2,f),t);
                    end
                end
                
                tst = predict(mdl,testsFeatures);
                
                for i = 1:length(tst)
                    predictedLabels(i) = tst(i);
                end
                featureWeights = mdl.Beta;
%                 %SVR
%                 ker = kernel('linear',2);
%                 algo = svr(ker);
%                 algo.epsilon = 0.00001;
%                 algo.optimizer = 'andre';
%                 algo.C = Inf;
%                 algo.balanced_ridge = 0;
%                 
%                 [tr a] = train(algo,trainData);
                
                %for t = 1:length(testLabels)
%                     for j = 1:nonZFeatures
%                         testsFeatures(j) = testRmat(featureList(j,1,f),featureList(j,2,f),t);
%                     end
%                     testsFeatures = zeros(nonZFeatures,length(testLabels));
%                     for j = 1:nonZFeatures
%                         testsFeatures(j,:) =  squeeze(testRmat(featureList(j,1,f),featureList(j,2,f),:))';
%                     end
%                     
%                     if size(testsFeatures,2)<10000
%                         testsData = data(testsFeatures',testLabels);
%                         testsTST = test(a,testsData);
%                         
%                         predictedLabels(:,f) = testsTST.X;
%                         clear testsFeatures testsData
%                         %end
%                     else
%                         ten_pct = round(linspace(1,length(testLabels),10));
%                         
%                         for t=2:10
%                             disp(['Testing ',num2str((t-1)*10),'%'])
%                             testsData = data(testsFeatures(:,ten_pct(t-1):ten_pct(t))',testLabels(ten_pct(t-1):ten_pct(t)));
%                             testsTST = test(a,testsData);
%                             predictedLabels(ten_pct(t-1):ten_pct(t),f) = testsTST.X;
%                             clear testsData
%                         end
%                         
%                     end
%                         featureWeights(1:nonZFeatures,f) = get_w(a);
            end
            R2 = corr(testLabels,predictedLabels).^2;
            performanceName = 'R2';
            consensusFeatureMat = featureWeights;
            
            
        else
            
            predictedTestLabels = zeros(length(testLabels),numSubjects,length(numFeatures));
            
            
            %Loop through and Leave out One for Cross Validation
            
            
            for i = 1:numSubjects
                %Leave out one subject
                disp(['Leave out Subject # ',num2str(i)])
                if i == 1
                    tempLabels = labels(2:end);
                    tempRmat = rmat(:,:,2:end);
                    testLabel = labels(1);
                elseif i == numSubjects
                    tempLabels = labels(1:numSubjects-1);
                    tempRmat = rmat(:,:,1:numSubjects-1);
                    testLabel = labels(numSubjects);
                else
                    tempLabels = cat(1,labels(1:i-1),labels(i+1:end));
                    tempRmat = cat(3,rmat(:,:,1:i-1),rmat(:,:,i+1:end));
                    testLabel = labels(i);
                end
                
                %Pick Top numFeatures from Training Data
                %[featureList(:,:,i,:),features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,noFeatSelect);
                [featureList,features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option);
                
                
                for f = 1:length(numFeatures)
                    
                    disp('Training and Testing SVR')
                    trainData = data(features(:,1:numFeatures(f),f),tempLabels);
                    
                    
                    %SVR
                    ker = kernel('linear',2);
                    algo = svr(ker);
                    algo.epsilon = 0.00001;
                    algo.optimizer = 'andre';
                    algo.C = Inf;
                    algo.balanced_ridge = 0;
                    
                    [tr a] = train(algo,trainData);
                    
                    %Extract features from Testing Data
                    testFeatures = zeros(1,numFeatures(f));
                    for j = 1:numFeatures(f)
                        testFeatures(j) = rmat(featureList(j,1,f),featureList(j,2,f),i);
                    end
                    testData = data(testFeatures,testLabel);
                    tst = test(a,testData);
                    
                    testsFeatures = zeros(length(testLabels),numFeatures(f));
                    for t = 1:length(testLabels)
                        for j = 1:numFeatures(f)
                            testsFeatures(t,j) = testRmat(featureList(j,1,f),featureList(j,2,f),t);
                        end
                    end
                    testsData = data(testsFeatures,testLabels);
                    testsTST = test(a,testsData);
                    
                    predictedTestLabels(:,i,f) = testsTST.X;
                    
                    %Get featureWeights
                    featureWeights(1:numFeatures(f),i,f) = get_w(a);
                    
                    %tst.X
                    predictedLabels(i,f) = tst.X;
                    
                    clear trainData tr a testData tst
                end
                clear tempLabels tempRmat testLabel
            end
            
            
            
            predictedLabels = predictedTestLabels;
            R2 = corr(testLabels,predictedLabels).^2;
            performanceName = 'predictedTestLabels';
        end
        
        % RESULTS
       

        for f = 1:length(numFeatures)
            results_struct(f).predictedLabels = predictedLabels(:,f);
            results_struct(f).featureList = featureList(:,:,:,f);
            results_struct(f).featureWeights = featureWeights(:,:,f);
            results_struct(f).R2 = R2(f);
        end

        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Paired SVM
    case 5  
        
        numPairs = numSubjects/2;
        
        %Loop through each pair and Leave Out Two for Cross Validation
        for i = 1:numPairs
            %Leave out one subject
            disp(['Leave out Pair # ',num2str(i)])
            if i == 1
                pair1tempLabels = labels(2:numPairs);
                pair1tempRmat = rmat(:,:,2:numPairs);
                pair1Label = labels(1);
                
                pair2tempLabels = labels(2+numPairs:end);
                pair2tempRmat = rmat(:,:,2+numPairs:end);
                pair2Label = labels(1+numPairs);
            elseif i == numPairs
                pair1tempLabels = labels(1:numPairs-1);
                pair2tempLabels = labels(numPairs+1:numSubjects-1);
                pair1tempRmat = rmat(:,:,1:numPairs-1);
                pair2tempRmat = rmat(:,:,numPairs+1:numSubjects-1);
                pair1Label = labels(numPairs);
                pair2Label = labels(numSubjects);
            else
                pair1tempLabels = [labels(1:i-1);labels(i+1:numPairs)];
                pair2tempLabels = [labels(numPairs+1:numPairs+i-1);labels(numPairs+i+1:end)];
                pair1tempRmat = cat(3,rmat(:,:,1:i-1),rmat(:,:,i+1:numPairs));
                pair2tempRmat = cat(3,rmat(:,:,numPairs+1:numPairs+i-1),rmat(:,:,numPairs+i+1:end));
                pair1Label = labels(i);
                pair2Label = labels(i+numPairs);
            end
            
            tempLabels = [pair1tempLabels; pair2tempLabels];
            tempRmat = cat(3,pair1tempRmat,pair2tempRmat);
            
            %Pick Top numFeatures from Training Data
            [featureList(:,:,i,:),features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,noFeatSelect,option);
            
            for f = 1:length(numFeatures)
                disp('Training and Testing SVM')
                trainData = data(features(:,1:numFeatures(f),f),tempLabels);
                
                [tr a] = train(svm,trainData);
                
                %Extract features from Testing Data
                pair1_testFeatures = zeros(1,numFeatures(f));
                pair2_testFeatures = zeros(1,numFeatures(f));
                for j = 1:numFeatures(f)
                    pair1_testFeatures(j) = rmat(featureList(j,1,i,f),featureList(j,2,i,f),i);
                    pair2_testFeatures(j) = rmat(featureList(j,1,i,f),featureList(j,2,i,f),i+numPairs);
                end
                
                pair1_testData = data(pair1_testFeatures,pair1Label);
                pair2_testData = data(pair2_testFeatures,pair2Label);
                
                pair1_tst = test(a,pair1_testData);
                pair2_tst = test(a,pair2_testData);
                
                pair1_tst.X
                pair2_tst.X
                %Get featureWeights
                featureWeights(1:numFeatures(f),i,f) = get_w(a);
                
                if pair1_tst.X == pair1Label
                    predictedLabels(i,f) = 1;
                else
                    predictedLabels(i,f) = 0;
                end
                
                if pair2_tst.X == pair2Label
                    predictedLabels(i+numPairs,f) = 1;
                else
                    predictedLabels(i+numPairs,f) = 0;
                end
                clear trainData tr a testData tst
            end
            clear tempLabels tempRmat testLabel
        end
        hitRate = sum(predictedLabels)./numSubjects.*100;
        performanceMeasure = hitRate;
        performanceName = 'hitRate';
        
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %% Multi-class SVM   
    case 6  
        
        hits = zeros(numSubjects,length(numFeatures));
        %Loop through and Leave out One for Cross Validation
        for i = 1:numSubjects
            %Leave out one subject
            disp(['Leave out Subject # ',num2str(i)])
            if i == 1
                tempLabels = labels(2:end,:);
                tempRmat = rmat(:,:,2:end);
                testLabel = labels(1,:);
            elseif i == numSubjects
                tempLabels = labels(1:numSubjects-1,:);
                tempRmat = rmat(:,:,1:numSubjects-1);
                testLabel = labels(numSubjects,:);
            else
                tempLabels = cat(1,labels(1:i-1,:),labels(i+1:end,:));
                tempRmat = cat(3,rmat(:,:,1:i-1),rmat(:,:,i+1:end));
                testLabel = labels(i,:);
            end
            
            %Pick Top numFeatures from Training Data
            [featureList,features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option);
            
            for f = 1:length(numFeatures)
                disp('Training and Testing MULTI-CLASS SVM')
                trainData = data(features(:,1:numFeatures(f),f),tempLabels);
                
                ker = kernel('linear',2);
                algo = mc_svm(ker);
                [tr a] = train(algo,trainData);
                
                %Extract features from Testing Data
                testFeatures = zeros(1,numFeatures(f));
                for j = 1:numFeatures(f)
                    testFeatures(j) = rmat(featureList(j,1,f),featureList(j,2,f),i);
                end
                
                testData = data(testFeatures,testLabel);
                
                tst = test(a,testData);
                
                
                %Get featureWeights
                featureWeights(1:numFeatures(f),i,f) = get_w(a);
                
                if tst.X == testLabel
                    hits(i,f) = 1;
                    predictedLabels(i,f) = find(tst.X==1);
                else
                    hits(i,f) = 0;
                    predictedLabels(i,f) = find(tst.X==1);
                end
            end
            clear trainData tr a testData tst tempLabels tempRmat testLabel
        end
        
        % RESULTS
        hitRate = sum(hits)./numSubjects.*100;
        performanceMeasure = hitRate;
        for f = 1:length(numFeatures)
            results_struct(f).predictedLabels = predictedLabels(:,f);
            results_struct(f).featureList = featureList(:,:,:,f);
            results_struct(f).featureWeights = featureWeights(:,:,f);
            results_struct(f).hitRate = hitRate(f);
        end
        performanceName = 'hitRate';
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Multi-class SVM with test subjects
    case 7 
        
        % If you only want to have a single model
        if option == 2
            predictedLabels = zeros(size(testLabels,1),length(numFeatures));
            
            %Single Model in the Training Set
            disp('Create Single Model in Training Set')
            [featureList(:,:,1,:),features] = variablefeatureSelector(rmat,labels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option); % select all features
            
            for f = 1:length(numFeatures)
                disp('Training and Testing MULTI-CLASS SVM')
                
                
                %trainData = data(features(:,1:numFeatures(f),f),labels);
                trainFeatures = features(:,1:numFeatures(f));
                
                mdl = fitcecoc(trainFeatures,labels);
                
                testFeatures = zeros(length(testLabels),numFeatures(f));
                 
                % Extract features from Testing Data
                for t = 1:size(testLabels,1)
                    for j = 1:numFeatures(f)
                        testFeatures(t,j) = testRmat(featureList(j,1,1,f),featureList(j,2,1,f),t);
                    end
                end
                    tst = predict(mdl,testFeatures);
                    
                    predictedLabels = tst;
                    featureWeights = mdl.BinaryLearners;
                    
                    hits = (tst==testLabels);
                    clear testsFeatures testsData
                
                
            end
            performanceMeasure = sum(hits)./size(testLabels,1);
            performanceName = 'testHitRate';
                        
            
        else
            
            predictedTestLabels = zeros(size(testLabels,1),numSubjects,length(numFeatures));
            
            
            %Loop through and Leave out One for Cross Validation
            
            
            for i = 1:numSubjects
                %Leave out one subject
                disp(['Leave out Subject # ',num2str(i)])
                if i == 1
                    tempLabels = labels(2:end,:);
                    tempRmat = rmat(:,:,2:end);
                    testLabel = labels(1,:);
                elseif i == numSubjects
                    tempLabels = labels(1:numSubjects-1,:);
                    tempRmat = rmat(:,:,1:numSubjects-1);
                    testLabel = labels(numSubjects,:);
                else
                    tempLabels = cat(1,labels(1:i-1,:),labels(i+1:end,:));
                    tempRmat = cat(3,rmat(:,:,1:i-1),rmat(:,:,i+1:end));
                    testLabel = labels(i,:);
                end
                
                %Pick Top numFeatures from Training Data
                %[featureList(:,:,i,:),features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,noFeatSelect);
                [featureList(:,:,i,:),features] = variablefeatureSelector(tempRmat,tempLabels,numFeatures,svr_check,svm_multi_check,noFeatSelect);
                
                
                for f = 1:length(numFeatures)
                    
                    disp('Training and Testing SVR')
                    trainData = data(features(:,1:numFeatures(f),f),tempLabels);
                    
                    
                    %SVR
                    ker = kernel('linear',2);
                    algo = mc_svm(ker);
                    [tr a] = train(algo,trainData);
                          
                    %Extract features from Testing Data
                    testFeatures = zeros(1,numFeatures(f));
                    for j = 1:numFeatures(f)
                        testFeatures(j) = rmat(featureList(j,1,f),featureList(j,2,f),i);
                    end
                    testData = data(testFeatures,testLabel);
                    tst = test(a,testData);
                    
                    for t = 1:size(testLabels,1)
                        testsFeatures = zeros(1,numFeatures(f));
                        for j = 1:numFeatures(f)
                            testsFeatures(j) = testRmat(featureList(j,1,f),featureList(j,2,f),t);
                        end
                        testsData = data(testsFeatures,testLabels);
                        testsTST = test(a,testsData);
                        
                        predictedTestLabels(t,i,f) = find(testsTST.X==1);
                        
                        
                        
                    end
                    
                    %Get featureWeights
                    %featureWeights(1:numFeatures(f),i,f) = get_w(a);
                    
                    %tst.X
                    predictedLabels(i,f) = find(tst.X==1);
                    
                    clear trainData tr a testData tst
                end
                clear tempLabels tempRmat testLabel
            end
                
            performanceMeasure = predictedTestLabels;
            performanceName = 'predictedTestLabels';
        end
        
            
        for f = 1:length(numFeatures)
            results_struct(f).predictedLabels = predictedLabels(:,f);
            results_struct(f).featureList = featureList(:,:,:,f);
            results_struct(f).featureWeights = featureWeights(:,:,f);
            results_struct(f).hitRate = performanceMeasure(f);
            
        end
end
%%%%%%%%%%%%%%%%%%% Return to Main Sequence %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
consensusFeatureList = {};

if ~noFeatSelect
    if option~=2
        if option ~=3
            % Consensusize Feature List
%             for f = 1:length(numFeatures)
% %                 consensusCount = 1; disp(['Consensusizing ',num2str(numFeatures(f)),' Feautres'])
% %                 u_featureList = unique(squeeze(featureList(:,:,1,f)),'rows');
% %                 for i = 1:length(u_featureList)
% %                     count = 1;
% %                     for n = 2:size(featureList,3)
% %                         if ismember(squeeze(u_featureList(i,:)),squeeze(featureList(:,:,n,f)),'rows')
% %                             count = count+1;
% %                         else
% %                             n = size(featureList,3);
% %                         end
% %                     end
% %                     
% %                     if count >= size(featureList,3) && u_featureList(i,1) > 0
% %                         consensusFeatureList{f,1}(consensusCount,:) = u_featureList(i,:);
% %                         consensusCount = consensusCount+1;
% %                     end
% %                 end
%                 
% 
%                 disp(['Number of consensus features = ',num2str(consensusCount-1)])
%             end
            
%             for f = 1:length(numFeatures)
%                 for i = 1:length(featureList(:,:,:,f))
%                     consensusFeatureMat(consensusFeatureList{f}(i,1),consensusFeatureList{f}(i,2),f) = 1;
%                     consensusFeatureMat(consensusFeatureList{f}(i,2),consensusFeatureList{f}(i,1),f) = 1;
%                 end
%             end
        end
    end
    
end



% FOR ALL Save things
%results_struct = struct('predictedLabels',predictedLabels,'consensusFeatureMat',consensusFeatureMat,'numFeatures',numFeatures,performanceName,performanceMeasure);

end
function [featureList, features] = variablefeatureSelector(rmat,labels,numFeatures,svr_check,svm_multi_check,noFeatSelect,option)

[numRows,numCols,numSubjects] = size(rmat);
testF = length(numFeatures);
if noFeatSelect % if you want to bypass feature selection
    nonZFeatureMat = sum(rmat,3)~=0;
    
    if numRows == numCols
        %featureList = zeros(numFeatures,2,1);
        %features = zeros(numSubjects,numFeatures);
        
        count = 1;
        for x = 1:numRows
            for y = x+1:numRows
                if nonZFeatureMat(x,y)~=0
                    featureList(count,:) = [x y];
                    count = count+1;
                end
            end
        end
        features = zeros(numSubjects,length(featureList));
        for n = 1:length(featureList)
            features(:,n) = squeeze(rmat(featureList(n,1),featureList(n,2),:));
        end
    else
              
        count = 1;
        for x = 1:numRows
        for y = 1:numCols
            featureList(count,:) = [x y];
            count = count+1;
        end
        end
        for n = 1:length(featureList)
            features(:,n) = squeeze(rmat(featureList(n,1),featureList(n,2),:));
        end
    end

%     for n = 1:numSubjects
%         if numRows == numCols
%             for i = 1:numRows
%                 rmat(i,i,n) = 0;
%             end
%         end
%         s_numFeatures = length(nonzeros(triu(squeeze(rmat(:,:,n)))));
%         [temp_fL(1:s_numFeatures,1) temp_fL(1:s_numFeatures,2)] = find(triu(squeeze(rmat(:,:,n))));
%         featureList = unique(cat(1,featureList,temp_fL),'rows');
%     end
%     
%     for n = 1: length(featureList)
%         features(:,n) = squeeze(rmat(featureList(n,1),featureList(n,2),:));
%     end
%     if length(featureList)~=numFeatures
%         featureList = cat(1,featureList,zeros(numFeatures-length(featureList),2));
%     end
else
    if option==3 % if you want to do random feature selection in each fold
        features = zeros(numSubjects,max(numFeatures),testF);
        featureList = zeros(max(numFeatures),2,testF);
        
        disp('Randomly selecting features in each fold...')
        
        if numRows == numCols % square so only take upper triangle
            possible_feat = (numRows*numCols-numRows)/2;
            count = 1;
            for i = 1:numRows
                for j=i+1:numCols
                    possible_feat_idx(count,:) = [i j];
                    count = count+1;
                end
            end
            
        else
            disp('WRITE SOMETHING FOR NON-SQUARE MATRICES')
        end
        
        for f = 1:testF
            featureList(1:numFeatures(f),:,f) = possible_feat_idx(randperm(possible_feat,numFeatures(f)),:);
            for n = 1:numSubjects
                for j = 1:numFeatures(f)
                features(n,j,f) = rmat(featureList(j,1,f),featureList(j,2,f),n);
                end
            end
        end
        
        else
    %if numFeatures
    %Reverse the effect of fisher transform on 1
    %rmat(rmat==Inf)=1;
    
    if svr_check
        if option ~=3 % if you don't want to randomly select features in each fold
            disp('Calculating correlation between ROI pair correlation and labels...')
            r_vs_labels = zeros(numRows,numCols);
            for i = 1:numRows
                temp = corr(squeeze(rmat(i,:,:))',labels);
                r_vs_labels(i,:) = temp;
                if numRows == numCols
                    r_vs_labels(i,i) = 0;
                end
            end
            r_vs_labels(isnan(r_vs_labels))=0;
            
            featRANK = abs(r_vs_labels);
        end
    else
        if ~svm_multi_check
            disp('T-testing...')
            rmat1 = rmat(:,:,find(labels>0));
            rmat2 = rmat(:,:,find(labels<0));
            [h,p,ci,stats] = ttest2(rmat1,rmat2,.01,'both','unequal',3);
            
            %Absolute value of t-stats
            featRANK = abs(stats.tstat);
        else
            disp('ANOVA testing...')
            cL = zeros(numSubjects,1);
            for i = 1:numSubjects
                cL(i) = find(labels(i,:)==1);
            end
            for i = 1:numRows
                for j = i+1:numCols
                    [p,tbl,stats] = anova1(rmat(i,j,:),cL,'off');
                    F_vs_labels(i,j) = abs(tbl{2,5});
                end
            end
            featRANK = abs(F_vs_labels);
        end
    end
    %List of top numFeatures ROI pairs
    featureList=zeros(max(numFeatures),2,testF);
    
        % If featRANK has too many zeros
        if sum(sum(featRANK==0)) >= numRows*numCols - numFeatures(end)
            disp('Too many 0 in t-test/corr vs. labels')
            for f = 1:length(numFeatures)
                featureList(1:numFeatures(f),:,f) = [randi(numRows,[numFeatures(f),1]),randi(numCols,[numFeatures(f),1])];
            end
        else
            for f = 1:testF
                start = 1;
                if f > 1
                    featureList(1:numFeatures(f-1),:,f) = featureList(1:numFeatures(f-1),:,f-1);
                    start = numFeatures(f-1)+1;
                end
                for i = start:numFeatures(f)
                    [num,idx] = max(featRANK(:));
                    %[num idx] = min(nonzeros(featRANK(:)));
                    [featureList(i,1,f) featureList(i,2,f)] = ind2sub(size(featRANK),idx);
                    featRANK(featureList(i,1,f),featureList(i,2,f)) = 0;
                    featRANK(featureList(i,2,f),featureList(i,1,f)) = 0;
                end
            end
        end
        %Create Feature Matrix
        features = zeros(numSubjects,max(numFeatures),testF);
        for f = 1:testF
            start=1;
            if f > 1
                features(:,1:numFeatures(f-1),f) = features(:,1:numFeatures(f-1),f-1);
                start = numFeatures(f-1)+1;
            end
            for i = start:numFeatures(f)
                for j = 1:numSubjects
                    if featureList(i,1,f) > 0
                        features(j,i,f) = squeeze(rmat(featureList(i,1,f),featureList(i,2,f),j));
                    end
                end
            end
        end
    end
end
end


function zmat = convertR2Z(rmat)
% The function of this script is to convert r-values to Z-values in a 2D Matrix

[numRows,numCols,numSubjects] = size(rmat);
zmat = zeros(numRows,numCols,numSubjects);
if numRows ==numCols
    numROIs = numRows;
end
for x = 1:numSubjects
    zCount = 1;
    % If square
    if numRows == numCols
        tempZmat = zeros((numROIs*numROIs-numROIs)/2,1);
        for i = 1:numROIs
            for j = i+1:numROIs
                tempZmat(zCount) = rmat(i,j,x);
                zCount = zCount + 1;
            end
        end
        %ztemp = cv_Fisher_r_to_Z_conv_2D(tempZmat);
        ztemp = atanh(tempZmat);
        Zcount = 1;
        for i = 1:numROIs
            for j = i+1:numROIs
                zmat(i,j,x) = ztemp(Zcount);
                zmat(j,i,x) = ztemp(Zcount);
                Zcount = Zcount+1;
            end
        end
    else % If not square
        tempZmat = zeros(numRows*numCols,1);
        for i = 1:numRows
            for j = 1:numCols
                tempZmat(zCount) = rmat(i,j,x);
                zCount = zCount + 1;
            end
        end
        ztemp = atanh(tempZmat);
        %ztemp = cv_Fisher_r_to_Z_conv_2D(tempZmat);
        Zcount = 1;
        for i = 1:numRows
            for j = 1:numCols
                zmat(i,j,x) = ztemp(Zcount);
                Zcount = Zcount +1;
            end
        end
    end
end

end

