% looking into specific network/subcortical correlations 
%  SMH, FP, DAN, DMN, SAL, amygdala, and hippocampus

% load data
load('cytokine_data/full_cytokines_corrmat_fullterm_latepreterm.mat')
load('/data/smyser/smyser1/wunder/eLABe/infomap_analyses/infomap_workingdir/correlations/adult_order.txt')
spreadsheet = readtable('cytokine_data/cytokines_data_sheet_fullterm_latepreterm.csv');

% reorder corrmat
subc_reorder = [334 341 335 342 336 343 337 344 338 345 339 346 340 347];
corrmat_reordered = corrmat([adult_order' subc_reorder], [adult_order' subc_reorder], :);

% hypothesized networks 
systemBOUNDS = [1 38; 39 46; 47 70; 71 78; 79 102; 103 141; 142 173; 174 213; 214 218; 219 222; 223 245; 246 286; 287 333; 334 347];
hypothesized_nets = {'SMH','FP','DAN','DMN','SAL','amygdala','hippocampus'};

hypothesized_nets_systemBOUNDS ( 1, : ) = systemBOUNDS(1,:);
hypothesized_nets_systemBOUNDS( 2,: ) = systemBOUNDS(5,:);
hypothesized_nets_systemBOUNDS( 3,: ) = systemBOUNDS(7,:);
hypothesized_nets_systemBOUNDS( 4,: ) = systemBOUNDS(12,:);
hypothesized_nets_systemBOUNDS( 5,: ) = systemBOUNDS(10,:);
hypothesized_nets_systemBOUNDS( 6,: ) = [340,341];
hypothesized_nets_systemBOUNDS( 7,: ) = [338,339];

numSubjects = size(corrmat_reordered, 3);
hypothesized_nets_per_subject = zeros(7,7,numSubjects);

% loop through each subject
for n = 1:numSubjects
    for s1 = 1:length(hypothesized_nets)
        for s2 = s1:length(hypothesized_nets)
            if s1==s2 % within network
                lt_within_net = tril(corrmat_reordered(hypothesized_nets_systemBOUNDS(s1,1):hypothesized_nets_systemBOUNDS(s1,2),hypothesized_nets_systemBOUNDS(s1,1): hypothesized_nets_systemBOUNDS(s1,2),n));
                hypothesized_nets_per_subject(s1,s1,n) = mean(nonzeros(lt_within_net));
            else % between network
                hypothesized_nets_per_subject(s1,s2,n) = mean(nonzeros(corrmat_reordered(hypothesized_nets_systemBOUNDS(s1,1):hypothesized_nets_systemBOUNDS(s1,2), hypothesized_nets_systemBOUNDS(s2,1):hypothesized_nets_systemBOUNDS(s2,2),n)));
            end
        end
    end
end

% calculate correlations with cytokines 
% IL6
hypothesized_nets_by_IL6 = zeros(7,7);  % initialize 7x7 matrix 
il6 = spreadsheet.il6_avg(:);           % IL-6 column vector

for s1 = 1:length(hypothesized_nets)
    for s2 = s1:length(hypothesized_nets)
        edge_vals = squeeze(hypothesized_nets_per_subject(s1,s2,:));  % N×1
        [hypothesized_nets_by_IL6(s1,s2), pvals_il6(s1,s2)] = corr(edge_vals, il6);
    end
end

% IL8
hypothesized_nets_by_IL8 = zeros(7,7);  % initialize 7x7 matrix 
il8 = spreadsheet.il8_avg(:);           % IL-8 column vector

for s1 = 1:length(hypothesized_nets)
    for s2 = s1:length(hypothesized_nets)
        edge_vals = squeeze(hypothesized_nets_per_subject(s1,s2,:));  % N×1
        [hypothesized_nets_by_IL8(s1,s2), pvals_il8(s1,s2)] = corr(edge_vals, il8);
    end
end

% IL10
hypothesized_nets_by_IL10 = zeros(7,7);  % initialize 7x7 matrix 
il10 = spreadsheet.il10_avg(:);           % IL-10 column vector

for s1 = 1:length(hypothesized_nets)
    for s2 = s1:length(hypothesized_nets)
        edge_vals = squeeze(hypothesized_nets_per_subject(s1,s2,:));  % N×1
        [hypothesized_nets_by_IL10(s1,s2), pvals_il10(s1,s2)] = corr(edge_vals, il10);
    end
end

% tnfa
hypothesized_nets_by_tnfa = zeros(7,7);  % initialize 7x7 matrix 
tnfa = spreadsheet.tnfa_avg(:);           % tnfa column vector

for s1 = 1:length(hypothesized_nets)
    for s2 = s1:length(hypothesized_nets)
        edge_vals = squeeze(hypothesized_nets_per_subject(s1,s2,:));  % N×1
        [hypothesized_nets_by_tnfa(s1,s2), pvals_tnfa(s1,s2)] = corr(edge_vals, tnfa);
    end
end

% looking at plots
figure; imagesc(mean(hypothesized_nets_per_subject, 3))
colorbar
caxis([-.2 .2])

figure; imagesc(hypothesized_nets_by_tnfa)
colorbar
caxis([-.1 .1])
