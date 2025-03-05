% -----------------------Samples----------------------------------------

% Fullterm - n=241 + LatePreterm - n=33
% Our cytonkines spreadsheet - n=283

% ----------------------Load in & Organize Data--------------------------
spreadsheet = readtable('cytokine_data/cytokines_data_sheet.csv');

% generating correlations matrices (run only once and then save as .mat file)
% num_subjects = height(spreadsheet);
% subjects = spreadsheet.modid;
% 
% no_mat = 0;
% for n = 1:num_subjects
%     if exist(['/data/smyser/smyser1/wunder/eLABe/gordon_pconns_plus_atlas_subcortical/full_mats/', subjects{n}, '_V1_a_gordon_parcel_plus_term_N50_eLABe_atlas_subcort.txt'], 'file')
%         temp_corrmat = load(['/data/smyser/smyser1/wunder/eLABe/gordon_pconns_plus_atlas_subcortical/full_mats/', subjects{n}, '_V1_a_gordon_parcel_plus_term_N50_eLABe_atlas_subcort.txt']);
%         corrmat(:,:,n) = temp_corrmat;
%     elseif exist(['/data/smyser/smyser1/wunder/eLABe/gordon_pconns_plus_atlas_subcortical/full_mats/', subjects{n}, '_V1_b_gordon_parcel_plus_term_N50_eLABe_atlas_subcort.txt'],'file')
%         temp_corrmat = load(['/data/smyser/smyser1/wunder/eLABe/gordon_pconns_plus_atlas_subcortical/full_mats/', subjects{n}, '_V1_b_gordon_parcel_plus_term_N50_eLABe_atlas_subcort.txt']);
%         corrmat(:,:,n) = temp_corrmat;
%     else 
%         no_mat = no_mat+1;
%         list_no_mat(no_mat) = n;
%     end
% end
% 
% save('full_cytokines_corrmat_MR.mat', 'corrmat')

% load in corrmat
load('cytokine_data/full_cytokines_corrmat.mat')

% -------------------ACTUAL MODELS vs. NULL MODELS----------------------

% For each sample/covariate list, we are going to build 100 models to
% predict IL6 that use different train/test split partitions. Those are
% our ACTUAL models and we will look at the distribution of the prediction
% accuracy. We're also going to generate null models for comparison. We are
% going to build 1000 null models, but we are doing 10 random scrambles of the IL6
% values and 100 train/test split partitions 

%IMPORTANT NOTE: we have to shuffle the random number generator, because my
%SVR scripts seed the random number generator!

% Eventually we will compare the actual and null model distributions 

% ----------------------IL6, ALL SAMPLES, NO COVARIATES---------------------
tic
ind_test_set_corrmat = 0; % we don't need this scenario so set to zero
ind_test_set_labels = 0; % same here
num_partitions = 100;
num_scrambles = 10;

%numCores = feature('numcores'); % was 36 
numWorkers = 12; % desired number of workers (based on cores available)
if isempty(gcp('nocreate'))  
    parpool('local', numWorkers);  % start a parallel pool 
end

% actual models
eLABE_predict_il6_noCOV = tenFOLD_svm_scripts_matlab_parallel(corrmat, spreadsheet.il6_avg, ind_test_set_corrmat, ind_test_set_labels, num_partitions);

% null models
parfor i = 1:num_scrambles
    rng('shuffle');  % shuffle the random number generator
    idx_rand = randperm(height(spreadsheet));
    fakeIL6 = spreadsheet.il6_avg(idx_rand);
    eLABE_predict_il6_NULL_noCOV{i} = tenFOLD_svm_scripts_matlab(corrmat, fakeIL6, ind_test_set_corrmat, ind_test_set_labels, num_partitions);
end

% Plot the results
[~,idx_sorted] = sort([eLABE_predict_il6_noCOV.R2]);

figure; plot(1:num_partitions,[eLABE_predict_il6_noCOV(idx_sorted).R2],'ro','MarkerSize',10)
hold on
for i = 1:num_scrambles
    plot(1:num_partitions,[eLABE_predict_il6_NULL_noCOV{i}(idx_sorted).R2],'k.')
end

set(gcf,'color','white')
box off

xlabel('Train/Test Partitions Sorted by Prediction of IL-6')
ylabel('R2')

axis([0 101 0 0.08])

title('IL6 PREDICTION, NO COVARIATES')
toc

% ----------------------IL6, ALL SAMPLES, ORIGINAL COVARIATES---------------------
% original covariates: 
% social disadvantage, PMA at scan, GA at birth, meanFD_pre, sex, mmr

tic
%%%% INPUTS %%%%%
num_partitions = 100; % only change if we want to run fewer partitions
num_scrambles = 10;


% change this list when we want to test different covariates
covariate_list = {'disadv_prenatal', 'mri_test_pma_scan_dob', 'screen_delivery_ga_weeks', 'child_sex'};

%%%%%% OUTPUTS %%%%%%%%
% change the suffix (i.e., origCOV) when different
% covariates/samples
[eLABE_predict_origCOV, eLABE_predictNULL_origCOV] = prediction_with_covariates_parallel(corrmat, spreadsheet.il6_avg, covariate_list, num_partitions, num_scrambles);

% add specific title 
title('FULLTERM ONLY, ORIG COVARIATES')
toc

