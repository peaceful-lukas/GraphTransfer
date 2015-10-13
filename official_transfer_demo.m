
local_env = 1;

% if ~local_env
%     cd /v9/code/GraphTransfer
% end

addpath 'util/'
addpath 'param/'
addpath 'ddcrp/'
addpath 'splme_new/'
addpath 'pgm/'
addpath 'pgm/RRWM/'
addpath 'official_transfer_splme_new/'
addpath 'official_transfer_splme_new/local_lme/'
addpath 'official_transfer_splme_new/util/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'


% dataset = 'pascal3d_pascal';
dataset = 'awa';
% dataset = 'voc';
method = 'splme_new';

DS = loadDataset(dataset, local_env);








%%%%%%% TRANSFER

clearvars -except DS local_env
% load ~/Desktop/exp_results/awa/splme_new_awa_5574.mat
load ~/Desktop/exp_results/awa/splme_new_awa_5166.mat % without || . ||F
dataset = 'awa';
method = 'splme_new';
local_env = 1;
param = result{1};
W = result{2};
U = result{3};
param.lambda_U_local = 10; % or 5
param.lr_U_local = 0.01;


[DS_official, U_train, U_test param_train, param_test] = officialSplitDataset(DS, U, param);
[train_clsnames, test_clsnames] = officialStringifyClasses(param_train, param_test);

[tPairs S] = officialTransferPairs(U_train, U_test, param_train, param_test);
str_tPairs = [train_clsnames(tPairs(:, 1))' test_clsnames(tPairs(:, 2))'];


param_test_0 = param_test;
param_test_prev = param_test;
param_test_new = param_test;

U_test_0 = U_test;
U_test_prev = U_test;
U_test_new = U_test;


coord_idx = officialVisualizePrototypes(U_test_new, param_test_new, [], [], test_clsnames);

for i=1:size(tPairs, 1)

    trClass = tPairs(i, 1);
    teClass = tPairs(i, 2);
    fprintf('\n\n\n================================ %s ---> %s ================================\n', train_clsnames{trClass}, test_clsnames{teClass});

    % Transfer
    scale_alpha = 1;
    [U_test_new, param_test_new, inferred_idx] = officialTransfer(U_train, U_test, trClass, teClass, scale_alpha, param_train, param_test);
    coord_idx = officialVisualizePrototypes(U_test_new, param_test_new, coord_idx, inferred_idx, test_clsnames);
    officialBargraphTransferResult(DS_official, W, U_test_new, param_test_new, U_test_prev, param_test_prev, test_clsnames);
    officialDispAccuracy(DS_official, W, U_test_new, param_test_new, U_test_prev, param_test_prev, test_clsnames);

    % Re-train
    U_test_new = officialLocalTrain(DS_official, W, U_test_new, param_test_new);
    officialDispAccuracy(DS_official, W, U_test_new, param_test_new, U_test_prev, param_test_prev, test_clsnames);
    officialBargraphTransferResult(DS_official, W, U_test_new, param_test_new, U_test_prev, param_test_prev, test_clsnames);
    
    U_test_prev = U_test_new;
    param_test_prev = param_test_new;

    pause;
end


officialDispAccuracy(DS_official, W, U_test_new, param_test_new, U_test_0, param_test_0, test_clsnames);
officialBargraphTransferResult(DS_official, W, U_test_new, param_test_new, U_test_0, param_test_0, test_clsnames);

