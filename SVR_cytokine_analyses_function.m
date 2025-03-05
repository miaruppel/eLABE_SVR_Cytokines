% taking existing code and reformatting into a function for testing various cytokines and model configurations 

function SVR_cytokine_analyses_function(cytokines_sheet_path, corrmat_path, cytokine_type, covs_Y_or_N)

% --- INPUTS ---
% cytokines_sheet_path - path to .csv file containing relevant cytonkine and covariate information
% corrmat_path - path to pre-generated (from cytokines sheet and gordon pconns) correlation matrix .mat file 
% cytokine_type - cytokine of interest in the analyses; four choices il6, il8, il10, tnfa (not case specific)
% covs_Y_or_N - to include or not include covariates in the model; expects a 'Y' or 'N' (not case specific)

% load in relevant data
spreadsheet = readtable(cytokines_sheet_path);
load(corrmat_path, 'corrmat')

% generate relevant cytokine information based on input
if strcmpi(cytokine_type, 'il6')
    cytokine = spreadsheet.il6_avg;
elseif strcmpi(cytokine_type, 'il8')
    cytokine = spreadsheet.il8_avg;
elseif strcmpi(cytokine_type, 'il10')
    cytokine = spreadsheet.il10_avg;
elseif strcmpi(cytokine_type, 'tnfa')
    cytokine = spreadsheet.tnfa_avg;
else 
    fprintf("INCORRECT CYTOKINE TYPE INPUT. Available options: 'il6', 'il8', 'il10', 'tnfa'")
end 

% run models
tic
if strcmpi(covs_Y_or_N, 'N') % no covariate models 
    ind_test_set_corrmat = 0; 
    ind_test_set_labels = 0; 
    num_partitions = 100;
    num_scrambles = 10;
    
    % parallel processing 
    numWorkers = 12; % desired number of workers (based on cores available)
    if isempty(gcp('nocreate'))  
        parpool('local', numWorkers);  % start a parallel pool 
    end

    % actual models
    eLABE_predict_noCOV = tenFOLD_svm_scripts_matlab_parallel(corrmat, cytokine, ind_test_set_corrmat, ind_test_set_labels, num_partitions);

    % null models
    parfor i = 1:num_scrambles
        rng('shuffle');  % shuffle the random number generator
        idx_rand = randperm(height(spreadsheet));
        fake = cytokine(idx_rand);
        eLABE_predict_NULL_noCOV{i} = tenFOLD_svm_scripts_matlab(corrmat, fake, ind_test_set_corrmat, ind_test_set_labels, num_partitions);
    end

    % plot the results
    [~,idx_sorted] = sort([eLABE_predict_noCOV.R2]);

    figure; plot(1:num_partitions,[eLABE_predict_noCOV(idx_sorted).R2],'ro','MarkerSize',10)
    hold on
    for i = 1:num_scrambles
        plot(1:num_partitions,[eLABE_predict_NULL_noCOV{i}(idx_sorted).R2],'k.')
    end

    set(gcf,'color','white')
    box off

    xlabel('Train/Test Partitions Sorted by Prediction')
    ylabel('R2')

    axis([0 101 0 0.08])

    plot_title = sprintf('%s PREDICTION, NO COVARIATES', cytokine_type);
    title(plot_title)
    toc

elseif strcmpi(covs_Y_or_N, 'Y') % including covariates in the model 
    num_partitions = 100; % only change if we want to run fewer partitions

    % parallel processing
    numWorkers = 12; % desired number of workers (based on cores available)
    if isempty(gcp('nocreate'))  
        parpool('local', numWorkers);  % start a parallel pool 
    end
    
    % original covariates: 
    % social disadvantage, PMA at scan, GA at birth, meanFD_pre, sex, mmr
    
    
    
    
    
    
else 
    fprintf("INCORRECT COVARIATE INPUT. Available options: 'Y' or 'N'")

end
end 
