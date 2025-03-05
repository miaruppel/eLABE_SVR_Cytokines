function [actual_models, null_models] = prediction_with_covariates_parallel(corrmat, spreadsheet,covariate_list,num_partitions)

ind_test_set_corrmat = 0; % we don't need this scenario so set to zero
ind_test_set_labels = 0; % same here
ind_test_set_cov = 0;
num_scrambles = 10;

covariates = zeros(length(spreadsheet.avg_il6),length(covariate_list)+1);
covariates(:,1) = ones(length(spreadsheet.avg_il6),1);

for i = 1:length(covariate_list)
    covariates(:,1+i) = spreadsheet.(covariate_list{i});
end

actual_models = tenFOLD_svm_scripts_covariates_matlab(corrmat,spreadsheet.avg_il6,covariates,ind_test_set_corrmat,ind_test_set_labels,ind_test_set_cov,num_partitions);

%numCores = feature('numcores');
numWorkers = 12; % desired number of workers (based on cores available)
if isempty(gcp('nocreate'))  
    parpool('local', numWorkers);  % start a parallel pool 
end

parfor i = 1:num_scrambles
    rng('shuffle');
    idx_rand = randperm(length(spreadsheet.avg_il6));
    fakeIL6 = spreadsheet.avg_il6(idx_rand);
    null_models{i} = tenFOLD_svm_scripts_covariates_matlab(corrmat,fakeIL6,covariates,ind_test_set_corrmat,ind_test_set_labels,ind_test_set_cov,num_partitions);
end

[~,idx_sorted] = sort([actual_models.R2]);

figure; plot(1:num_partitions,[actual_models(idx_sorted).R2],'ro','MarkerSize',10)
hold on
for i = 1:num_scrambles
    plot(1:num_partitions,[null_models{i}(idx_sorted).R2],'k.')
end

set(gcf,'color','white')
box off

xlabel('Train/Test Partitions Sorted by Prediction of IL-6')
ylabel('R2')

axis([0 101 0 0.08])
