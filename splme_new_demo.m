
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
addpath 'transfer/'
addpath 'transfer/local_lme/'
addpath 'transfer/util/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'


% dataset = 'pascal3d_pascal';
dataset = 'awa';
% dataset = 'voc';
method = 'splme_new';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U param] = splme(DS, param, local_env);
splme_new;




%%%%%%% TRANSFER

clearvars -except DS local_env
load ~/Desktop/exp_results/awa/splme_new_awa_5574.mat
dataset = 'awa';
method = 'splme_new';
local_env = 1;
param = result{1};
W = result{2};
U = result{3};
coord_idx = visualizeBoth(DS, W, U, param, [], []);

clsnames = stringifyClasses(param.dataset);
[tPairs str_tPairs scores S] = transferPairs(U, param);
% [tPairs str_tPairs] = setTransferDirections(tPairs, str_tPairs, perClassScores);



param0 = param;
param_new = param;
param_prev = param;

U0 = U;
U_prev = U;
U_new = U;

for i=1:size(tPairs, 1)

    % Transfer direction : c1 ------->>> c2
    c1 = tPairs(i, 1);
    c2 = tPairs(i, 2);
    fprintf('\n\n\n================================ %s ---> %s ================================\n', clsnames{c1}, clsnames{c2});

    scale_alpha = 1;
    [U_new, param_new, matched_pairs, inferred_idx, trainTargetClasses, score_GM] = transfer(DS, W, U_new, W, U_prev, c1, c2, scale_alpha, param_new, param_prev);

    % Visualize
    % coord_idx = visualizePrototypes(U_new, param_new, coord_idx, inferred_idx);
    % pause;

    if length(trainTargetClasses) > 0
        trainTargetClasses
        % % Locally train
        [U_new param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses);

        % Result
        % transfer_dispAccuracies(DS, W, U_new, W, U0, param_new, param0);

        % Visualize
        visualize = 1;
        % result = findExamples('gotTrueAfterTransfer', c2, DS, W, U, param, W, U_new, param_new, visualize);
        coord_idx = visualizePrototypes(U_new, param_new, coord_idx, inferred_idx);
        % pause;

        bargraphTransferResult(DS, W, U_new, param_new, U_prev, param_prev);
    else
        fprintf('No classes to be learned locally.\n');
    end

    [C_prev pr_labels_prev] = getConfusionMatrix(DS, W, U_prev, param_prev);
    [C_new pr_labels_new] = getConfusionMatrix(DS, W, U_new, param_new);

    opts.type = 'predicted';
    opts.numRows = 5;
    opts.numCols = 5;
    opts.title = 'Before Transfer';
    visualizeConfusionMat(DS, C_prev, pr_labels_prev, c2, opts);

    opts.title = 'After Transfer';
    visualizeConfusionMat(DS, C_new, pr_labels_new, c2, opts);

    C_prev - C_new

    U_prev = U_new;
    param_prev = param_new;

    pause;
end

