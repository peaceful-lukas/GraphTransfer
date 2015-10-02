% GRAPH MATCHING SCORES

MatchingScores = matchingScores(U, param);






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars -except DS local_env
load ~/Desktop/exp_results/pascal3d_pascal/splme_pascal3d_pascal_7185.mat
param = result{1}
W = result{2};
U = result{3};

dataset = 'pascal3d_pascal';
method = 'splme';
local_env = 1;
% DS = loadDataset(dataset, local_env);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
coord_idx = [];
[tPairs str_tPairs scores S] = transferPairs(U, param)

param0 = param;
param_new = param;
W0 = W;
W_new = W;
U0 = U;
U_new = U;

for i=1:12
    pair = tPairs(i, :);

    c1 = pair(1);
    c2 = pair(2);
    scale_alpha = 1;
    [U_new, param_new, matched_pairs, inferred_idx, trainTargetClasses, score_GM] = transfer(DS, W_new, U_new, W0, U0, c1, c2, scale_alpha, param_new, param0);

    % coord_idx = visualizePrototypes(U_new, param_new, coord_idx, matched_pairs(:, 1));
    % pause;
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clearvars -except DS local_env
load ~/Desktop/exp_results/awa/splme_awa_6833.mat
param = result{1};
W = result{2};
U = result{3};

dataset = 'awa';
method = 'splme';
local_env = 1;


coord_idx = [];
clsnames = stringifyClasses(param.dataset);
[tPairs str_tPairs scores S] = transferPairs(U, param)

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
        coord_idx = visualizePrototypes(U_new, param_new, coord_idx, inferred_idx);
        pause;
    end

    result = findExamples('gotTrueAfterTransfer', c2, DS, W, U, param, W_new, U_new, param_new);
    for r=1:length(result)
        imshow(DS.DI{result(r)});
        pause;
    end
end







param_new = param0;
U_new = U0;
param_new.lambda_U_local = 0.1; % or 5