function [actual_models, null_models] = prediction_with_covariates_parallel(corrmat, cytokine, covariate_list, num_partitions, num_scrambles)

global spreadsheet

ind_test_set_corrmat = 0; % we don't need this scenario so set to zero
ind_test_set_labels = 0; % same here
ind_test_set_cov = 0;

covariates = zeros(length(cytokine),length(covariate_list)+1);
covariates(:,1) = ones(length(cytokine),1);

for i = 1:length(covariate_list)
    covariates(:,1+i) = spreadsheet.(covariate_list{i});
end

% actual model
actual_models = tenFOLD_svm_scripts_covariates_matlab(corrmat, cytokine, covariates, ind_test_set_corrmat, ind_test_set_labels, ind_test_set_cov, num_partitions);

% parallel processing
numWorkers = 12; % desired number of workers (based on cores available)
if isempty(gcp('nocreate'))  
    parpool('local', numWorkers);  % start a parallel pool 
end

% null models
parfor i = 1:num_scrambles
    rng('shuffle');
    idx_rand = randperm(height(spreadsheet));
    fake = cytokine(idx_rand);
    null_models{i} = tenFOLD_svm_scripts_covariates_matlab(corrmat, fake, covariates, ind_test_set_corrmat, ind_test_set_labels, ind_test_set_cov, num_partitions);
end

end 

