function results = tenFOLD_svm_scripts_covariates_matlab(corrmat, labels, covariates, ind_test_set_corrmat, ind_test_set_labels, ind_covariates, num_iter)

% INPUTS
% corrmat - numROIs X numROIs X numSubjects
% labels - what you are trying to predict with FC (e.g., age) - numSubjects
    % X 1
% covariates - to be regressed out of both FC and labels separately
    % should be a cell containing numCOV x numSubjects
% ind_test_set_corrmat - numROIs x numROIs X numTESTSubjects or 0 if you
    %aren't going to test an independent set
% ind_test_set_labels - numTESTSubjects X 1 or 0 if you aren't going to
    % test an independent  set
% ind_covariates - numCov X numTESTSubjects or 0 if you aren't going to
    % test an independent  set
% num_iter - the number of train/test partitions you want to run
%addpath('/data/cn5/ashley/')    

if size(covariates,1)>size(covariates,2)
    covariates = covariates';
end
%IDENTIFY ONLY SUBJECTS w/ no NaNs
covariates_NaN = sum((isnan(covariates)));
idx_nonNaN = find((isnan(labels)+covariates_NaN'~=0)==0);

numSubjects = length(labels(idx_nonNaN));

if size(covariates,1) == 1
    covariates = covariates';
end

% CREATE THE 10-FOLD TRAINING and TEST SETS
rng(42);

test_set_extra = mod(numSubjects,10);
test_set_size = zeros(10,1);
for i = 1:10
    if i<=test_set_extra
        test_set_size(i) = floor(numSubjects/10)+1;
    else
        test_set_size(i) = floor(numSubjects/10);
    end
end

for n = 1:num_iter % number of 10-fold interations--for now 1
    subject_order(:,n) = randperm(numSubjects);
    subs_before = 0;
    for cv = 1:10
        test_set{n,cv} = subject_order(1+subs_before:subs_before+test_set_size(cv),n);
        train_set{n,cv} = setdiff(1:numSubjects,test_set{n,cv});
        subs_before = subs_before+test_set_size(cv);
    end
end

numIterations = size(train_set,1);
if size(labels,1)==1
    labels = labels';
end

if size(ind_test_set_labels,1)==1
    ind_test_set_labels = ind_test_set_labels';
end

for n = 1:numIterations
    disp(['Iter #',num2str(n)])
    predictedLabels = zeros(length(labels),1);
    regressedLabels = zeros(length(labels),1);
    %predictedTestLabels = zeros(length(ind_test_set_labels),10);
    hitRate = zeros(10,1);
    for cv = 1:10
        %disp(['FOLD #',num2str(cv)])
        
        % regress covariates from the train_set FC
        disp('regressing....')
        corrmat_regress_train = zeros(size(corrmat,1),size(corrmat,1),length(train_set{n,cv}));
        corrmat_regress_test = zeros(size(corrmat,1),size(corrmat,1),length(test_set{n,cv}));
        corrmat_regress_ind_test = zeros(size(corrmat,1),size(corrmat,1),length(ind_test_set_labels));
        for i = 1:size(corrmat,1)
            for j = i+1:size(corrmat,1)
                [~,~,corrmat_regress_train(i,j,:)] = regress(squeeze(corrmat(i,j,idx_nonNaN(train_set{n,cv}))),covariates(:,idx_nonNaN(train_set{n,cv}))');
                corrmat_regress_train(j,i,:) = corrmat_regress_train(i,j,:);
                
                [~,~,corrmat_regress_test(i,j,:)] = regress(squeeze(corrmat(i,j,idx_nonNaN(test_set{n,cv}))),covariates(:,idx_nonNaN(test_set{n,cv}))');
                corrmat_regress_test(j,i,:) = corrmat_regress_test(i,j,:);
                if size(ind_test_set_corrmat,1) > 1
                [~,~,corrmat_regress_ind_test(i,j,:)] = regress(squeeze(ind_test_set_corrmat(i,j,:)),ind_covariates);
                corrmat_regress_ind_test(j,i,:) = corrmat_regress_ind_test(i,j,:);
                end
            end
        end
                
        % regress covariates from the train_set labels
        [~,~,labels_regress_train] = regress(labels(idx_nonNaN(train_set{n,cv})),covariates(:,idx_nonNaN(train_set{n,cv}))');
        %labels_regress_train = labels(train_set{n,cv});
        [~,~,labels_regress_test] = regress(labels(idx_nonNaN(test_set{n,cv})),covariates(:,idx_nonNaN(test_set{n,cv}))');
        %labels_regress_test = labels(test_set{n,cv});
        if size(ind_test_set_corrmat,1) > 1
        [~,~,labels_regress_ind_test] = regress(ind_test_set_labels,ind_covariates);
        end
        % regress covariates from the test_set FC
        
        % regress covariate from the test_set labels
        
        
        temp = svm_scripts_v2021_matlab(corrmat_regress_train,labels_regress_train,0,corrmat_regress_test,labels_regress_test,2);
        
        % fix this order
        for i = 1:length(test_set{n,cv})
            predictedLabels(idx_nonNaN(test_set{n,cv}(i))) = temp.predictedLabels(i);
            regressedLabels(idx_nonNaN(test_set{n,cv}(i))) = labels_regress_test(i);
            
        end
        if numel(unique(predictedLabels))<=3
            hitRate(cv) = temp.hitRate;
        end
        
        if size(ind_test_set_corrmat,1) > 1 %  INDEPENDENT TEST SET
            temp = svm_scripts_v2021_matlab(corrmat_regress_train,labels_regress_train,0,corrmat_regress_ind_test,labels_regress_ind_test,2);
            
            predictedTestLabels(:,cv) = temp.predictedLabels;
            if numel(unique(predictedTestLabels))<=3
                testHitRate(cv) = temp.hitRate;
            end
            
            
        end
        fW = zeros(size(corrmat,1));
        for i = 1:size(temp.featureList,1)
            fW(temp.featureList(i,1,1),temp.featureList(i,2,1)) = temp.featureWeights(i,1);
            fW(temp.featureList(i,2,1),temp.featureList(i,1,1)) = temp.featureWeights(i,1);
        end
        featureWeights{cv} = fW;
       
    end
results(n).predictedLabels = predictedLabels;
results(n).regressedLabels = regressedLabels;
results(n).regressedR2 = corr(regressedLabels(idx_nonNaN),predictedLabels(idx_nonNaN)).^2;
results(n).featureWeights = featureWeights;
if numel(unique(predictedLabels))>3
    results(n).R2 = corr(labels(idx_nonNaN),predictedLabels(idx_nonNaN)).^2;
else
    results(n).hitRate = mean(hitRate);

end
results(n).N = length(idx_nonNaN);

if size(ind_test_set_corrmat,1)>1
    results(n).predictedTestLabels = predictedTestLabels;
    results(n).regressedTestLabels = labels_regress_ind_test;
    if numel(unique(predictedLabels))>2
        results(n).testR2 = (corr(ind_test_set_labels,mean(predictedTestLabels,2))).^2;
        results(n).regressedTestR2 = (corr(labels_regress_ind_test,mean(predictedTestLabels,2))).^2;
    else
        results(n).testHitRate = mean(testHitRate);
        
    end
end
    
    




end