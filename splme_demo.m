
local_env = 1;

% if ~local_env
%     cd /v9/code/GraphTransfer
% end

addpath 'util'
addpath 'param'
addpath 'ddcrp'
addpath 'splme'
addpath 'pgm'
addpath 'pgm/RRWM/'
addpath 'transfer'
addpath 'transfer/local_lme/'
addpath 'transfer/util/'
addpath 'tool/vis/distinguishable_colors/'
% addpath(genpath(pwd));

dataset = 'pascal3d_pascal';
% dataset = 'awa';
method = 'splme';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U param] = splme(DS, param, local_env);
splme;




%%%%%%% TRANSFER

clearvars -except DS local_env
load ~/Desktop/exp_results/awa/splme_awa_6881.mat
dataset = 'awa';
method = 'splme';
local_env = 1;
param = result{1};
W = result{2};
U = result{3};
perClassScores = result{4};
coord_idx = [];

clsnames = stringifyClasses(param.dataset);
[tPairs str_tPairs scores S] = transferPairs(U, param);
[tPairs str_tPairs] = setTransferDirections(tPairs, str_tPairs, perClassScores);

param0 = param;
param_new = param;
W0 = W;
W_new = W;
U0 = U;
U_new = U;


for i=1:size(tPairs, 1)

    % Transfer direction : c1 ------->>> c2
    if param_new.numPrototypes(tPairs(i, 1)) > param_new.numPrototypes(tPairs(i, 2))
        c1 = tPairs(i, 1);
        c2 = tPairs(i, 2);
    else
        c1 = tPairs(i, 2);
        c2 = tPairs(i, 1);
    end

    fprintf('\n\n\n================================ %s ---> %s ================================\n', clsnames{c1}, clsnames{c2});

    scale_alpha = 1;
    [U_new, param_new, matched_pairs, inferred_idx, trainTargetClasses, score_GM] = transfer(DS, W_new, U_new, W0, U0, c1, c2, scale_alpha, param_new, param0);

    % Visualize
    % coord_idx = visualizePrototypes(U_new, param_new, coord_idx, inferred_idx);
    % pause;

    if length(inferred_idx) > 0
        % % Locally train
        [U_new param_new] = local_train(DS, W_new, U_new, param_new, trainTargetClasses);

        % Result
        transfer_dispAccuracies(DS, W_new, U_new, W0, U0, param_new, param0);

        % Visualize
        visualize = 1;
        result = findExamples('gotTrueAfterTransfer', c2, DS, W, U, param, W_new, U_new, param_new, visualize);
        coord_idx = visualizePrototypes(U_new, param_new, coord_idx, inferred_idx);
        pause;
    end
end

