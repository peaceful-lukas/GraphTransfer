
local_env = 1;

addpath 'util/'
addpath 'param/'
addpath 'ddcrp/'
addpath 'splme_dist/'
addpath 'pgm/'
addpath 'pgm/RRWM/'
addpath 'transfer_splme_dist/'
addpath 'transfer_splme_dist/local_lme/'
addpath 'transfer_splme_dist/util/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'


dataset = 'voc';
% dataset = 'awa';
method = 'splme_dist';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U param] = splme_dist(DS, param, local_env);
splme_dist;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRANSFER PROTOTYPES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except DS local_env
% load ~/Desktop/exp_results/voc/splme_dist_voc_7378.mat % dimension 30 maximum accuracy
load ~/Desktop/exp_results/voc/splme_dist_voc_7513.mat % dimension 100 maximum accuracy
dataset = 'voc';
method = 'splme_dist';
local_env = 1;
param = result{1};
W = result{2};
U = result{3};

coord_idx = visualizePrototypes(U, param, [], []);

param.lambda_W_local = 0.001;
param.lambda_U_local = 0.00001; % or 5
param.lr_U_local = 0.1;

clsnames = stringifyClasses(param.dataset);
[tPairs str_tPairs scores S] = transferPairs(U, param);
% [tPairs str_tPairs] = setTransferDirections(tPairs, str_tPairs, perClassScores);


new_tPairs = zeros(3, 2);
new_tPairs(1, :) = tPairs(1, [1 2]);
new_tPairs(2, :) = tPairs(4, [1 2]);
new_tPairs(3, :) = tPairs(8, [2 1]);
% tPairs(2:4, :) = tPairs(2:4, [2 1]);
% str_tPairs(2:4, :) = str_tPairs(2:4, [2 1]);
tPairs = new_tPairs;


% visualizeTargetClassesBoth(DS, W, U, param, coord_idx, [1 2 3], 'test');
visualizeTargetExamplesWithAllPrototypes(DS, W, U, param, coord_idx, [1 2 3], 'test');
transfer_dispAccuracies(DS, W, U, W, U, param, param);



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

    U_prev = U_new;
    param_prev = param_new;

    % Visualize
    coord_idx = visualizePrototypes(U_new, param_new, coord_idx, inferred_idx);
    % pause;

    % if length(trainTargetClasses) > 0
        % clsnames(trainTargetClasses)'
        % Locally train
        % [U_new param_new] = local_train(DS, W, U_new, param_new, trainTargetClasses);

    %     % Result
        % transfer_dispAccuracies(DS, W, U_new, W, U_prev, param_new, param_prev);

    %     % Visualize
    %     % visualize = 1;
    %     % result = findExamples('gotTrueAfterTransfer', c2, DS, W, U, param, W, U_new, param_new, visualize);
    %     % coord_idx = visualizePrototypes(U_new, param_new, coord_idx, inferred_idx);
    %     % pause;

        % bargraphTransferResult(DS, W, U_new, param_new, U_prev, param_prev);
    % else
    %     fprintf('No classes to be learned locally.\n');
    % end

    % [C_prev pr_labels_prev] = getConfusionMatrix(DS, W, U_prev, param_prev);
    % [C_new pr_labels_new] = getConfusionMatrix(DS, W, U_new, param_new);

    % opts.type = 'predicted';
    % opts.numRows = 5;
    % opts.numCols = 5;
    % opts.title = 'Before Transfer';
    % visualizeConfusionMat(DS, C_prev, pr_labels_prev, c2, opts);

    % opts.title = 'After Transfer';
    % visualizeConfusionMat(DS, C_new, pr_labels_new, c2, opts);

    % C_prev - C_new

    U_prev = U_new;
    param_prev = param_new;

    pause;
end



