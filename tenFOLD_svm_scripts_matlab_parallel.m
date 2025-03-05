function results = tenFOLD_svm_scripts_matlab_parallel(corrmat, labels, ind_test_set_corrmat, ind_test_set_labels, num_iter)

% INPUTS

% corrmat - numROIs X numROIs X numSubjects
% labels - what you are trying to predict with FC - numSubjects X 1 - Ok if you included NaNs
        % could be BINARY: e.g., insurance status (0,1)
        % could be CATEGORICAL (up to 20): e.g., education (1,2,3,4)
        % could be CONTINUOUS: e.g., age (36.4-45.2 weeks)

% ind_test_set_corrmat - numROIs x numROIs X numTESTSubjects or 0 if you
%aren't going to test an independent set
% ind_test_set_labels - numTESTSubjects X 1 or 0 if you aren't going to
% test an independent  set
% num_iter - number of train/test partitions

% OUTPUTS - STRUCTURE w/ these variables

% predictedLabels - numSubjects X 1
        % for the SVM case: 1 if correct, 0 if incorrect
        % for the SVR case: predicted continuous label
        % for the MC-SVM case: class # a subject was categorized as (e.g., 3 out 1,2,3,4,5)
% R2 - 1 value; only for the SVR case, correlation between the true/predicted labels ^2
% hitRate - 1 value
        % for the SVM case: % of subjects correctly classified (if equal # in each class,  chance = 50%)
        % for the MC-SVM: % of subjects correctly classified (chance is < 50% and depends on #/size of classes)
% featureWeights
        % for the SVM/SVR case: {10 x 1} cell with numROIs x numROIS
            % each cell is the featureWeights from 1 fold of cross-validation
            % each matrix is how all the included FC was weighted in training 
                % featurWeights can be visualized w/ BLAH file I made for Jeanette
        % for the MC-SVM case: {10 x 1} cell with another layer of cells
            % feature weights for every possible combo of classes (e.g., 1vs2, 1vs3, 1vs4, 2vs3, 2vs4, 3vs4) 
% N - 1 value; number of subjects w/o NaNs
% predictedTestLabels - numTESTsubjects x 1
        % for the SVM case: 1 if correct, 0 if incorrect
        % for the SVR case: predicted continuous label
        % for the MC-SVM case: class # a subject was categorized as (e.g., 3 out 1,2,3,4,5)
% testR2 - 1 value; only for the SVR case, correlation between the true/predicted labels ^2 in the independent dataset
% testHitRate - 1 value for the independent datset
        % for the SVM case: % of subjects correctly classified (if equal # in each class,  chance = 50%)
        % for the MC-SVM: % of subjects correctly classified (chance is < 50% and depends on #/size of classes)


% DEPENDENCIES:
% svm_scripts_v2021_matlab.m

% USAGE:
% eLABE_predictAGE = tenFOLD_svm_scripts_matlab(corrmat,PMA,0,0);  % SVR on a single dataset
% predictAGE_trainELABE_testCUDDEL = tenFOLD_svm_scripts_matlan(corrmat_eLABE,PMA_eLABE,corrmat_CUDDEL,PMA_CUDDEL); % SVR w/ a independent test set
% eLABE_predictINSUR = tenFOLD_svm_scripts_matlab(corrmat,insur,0,0); % SVM on a single dataset
% eLABE_predictEDU = tenFOLD_svm_scripts_matlab(corrmat,edu,0,0); % MC-SVM on a single datasetn


% IDENTIFY ONLY SUBJECTS w/ no NaNs
idx_nonNaN = find(~isnan(labels)==1);
numSubjects = length(labels(idx_nonNaN));


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
numIterations = num_iter;

if size(labels,1)==1
    labels = labels';
end

if size(ind_test_set_labels,1)==1
    ind_test_set_labels = ind_test_set_labels';
end

% WHAT TYPE OF SVM ARE WE DOING HERE?
% binary SVM check
binary_svm = false;
if (numel(unique(labels(idx_nonNaN)))==2)
    binary_svm = true;
end
% continuous SVR check
cont_svr = false;
if (numel(unique(labels(idx_nonNaN))) >=4)
    cont_svr = true;
end
% multi-class SVM check
multi_class = false;
if (numel(unique(labels(idx_nonNaN))) > 2) && (numel(unique(labels(idx_nonNaN))) < 4)
    multi_class = true;
end

% ENTER INTO FOLDS OF TRAINING/TESTING
numFolds = 10;

% Preallocate `futures` as an array of FevalFuture objects for the cross-validation folds
futures = parallel.FevalFuture.empty(numFolds, 0);

for n = 1:numIterations
    disp(['Iter #', num2str(n)])
    
    % Initialize variables for this iteration
    predictedLabels = nan(length(labels), 1);
    predictedTestLabels = nan(length(ind_test_set_labels), numFolds);
    hitRate = zeros(numFolds, 1);

    for cv = 1:numFolds
        disp(['FOLD #', num2str(cv)])
        
        % Asynchronously run the function using parfeval
        futures(cv) = parfeval(@svm_scripts_v2021_matlab, 1, corrmat(:,:,idx_nonNaN(train_set{n,cv})), labels(idx_nonNaN(train_set{n,cv})), 0, corrmat(:,:,idx_nonNaN(test_set{n,cv})), labels(idx_nonNaN(test_set{n,cv})), 2);
    end

    % Collect results as each fold completes
    for cv = 1:numFolds
        temp = fetchOutputs(futures(cv)); % Retrieve result for each fold

        % REORGANIZE PREDICTED LABELS FROM TEST SET BACK into the TOTAL SET
        for i = 1:length(idx_nonNaN(test_set{n,cv})) 
            predictedLabels(idx_nonNaN(test_set{n,cv}(i))) = temp.predictedLabels(i);
        end

        % Process independent test set if applicable
        if size(ind_test_set_corrmat, 1) > 1
            temp = svm_scripts_v2021_matlab(corrmat(:,:,idx_nonNaN(train_set{n,cv})), labels(idx_nonNaN(train_set{n,cv})), 0, ind_test_set_corrmat, ind_test_set_labels, 2);
            predictedTestLabels(:, cv) = temp.predictedLabels;
            if binary_svm || multi_class
                testHitRate(cv) = temp.hitRate;
            end
        end
    end

    % Save results for this iteration
    results(n).predictedLabels = predictedLabels;
    if cont_svr
        results(n).R2 = corr(labels(idx_nonNaN), predictedLabels(idx_nonNaN)).^2;
        results(n).N = length(idx_nonNaN);
    end
    if multi_class
        results(n).hitRate = mean(predictedLabels(idx_nonNaN) == labels(idx_nonNaN));
        results(n).N = length(idx_nonNaN);
    end
    if binary_svm
        results(n).hitRate = mean(predictedLabels(idx_nonNaN));
        results(n).N = length(idx_nonNaN);
    end

    % Independent test set results
    if size(ind_test_set_corrmat, 1) > 1
        results(n).predictedTestLabels = predictedTestLabels;
        if cont_svr
            results(n).testR2 = (corr(ind_test_set_labels, mean(predictedTestLabels, 2))).^2;
        end
        if binary_svm
            results(n).testHitRate = mean(testHitRate);
        end
        if multi_class
            results(n).testHitRate = mean(predictedTestLabels == ind_test_set_labels);
        end
    end
end