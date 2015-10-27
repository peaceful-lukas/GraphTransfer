local_env = 1;
if ~local_env, cd '/v9/code/GraphTransfer/'; end

addpath 'util/'
addpath 'param/'
addpath 'sc/'
addpath 'clustering/'
addpath 'graph/'
addpath 'splme_cvpr/'
addpath 'tool/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'

% dataset = 'awa50';
% dataset = 'awa50_pca500';
% dataset = 'awa10_pca500';

% dataset = 'voc_pca500';
% dataset = 'voc4_pca500';

dataset = 'voc_high';

% dataset = 'voc_high_pca500';
% dataset = 'voc4_high_pca500';

method = 'splme_cvpr';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);
param.numInstancesPerClass = hist(DS.DL, 4)';

splme_cvpr;












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRANSFER PROTOTYPES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except DS local_env
load ~/Desktop/exp_results/voc_high/splme_cvpr_voc_high_8649.mat % dimension 100 maximum accuracy
dataset = 'voc_high';
method = 'splme_cvpr';
local_env = 1;
param = result{1};
W = result{2};
U = result{3};

coord_idx = visualizePrototypes(U, param, [], []);

param.lambda_W_local = 0.001;
param.lambda_U_local = 0.00001; % or 5
param.lr_U_local = 0.1;

% clsnames = stringifyClasses(param.dataset);
clsnames = {'bicycle', 'motorbike', 'bus', 'train'};
% [tPairs str_tPairs scores S] = transferPairs(U, param);
tPairs = [1 2; 3 4];
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



