
clearvars -except DS local_env
load ~/Desktop/awa_lme_result/splme_awa_6510.mat
dataset = 'awa_lme_result';
method = 'splme';
local_env = 1;
param = result{1};
W = result{2};
U = result{3};
coord_idx = [];



% ------------- separate train/test classes and prototypes
test_classes = [6, 14, 15, 18, 24, 25, 34, 39, 42, 48];
test_classes = test_classes';

train_classes = 1:50;
train_classes(find(ismember(train_classes, test_classes))) = [];
train_classes = train_classes';

trainPrototypes = getTargetPrototypeIndices(train_classes, param);
testPrototypes = getTargetPrototypeIndices(test_classes, param);

% ------------- Class similarities between official train / test classes
% tPairs = ( c_train, c_test )
% transfer direction: c_train --> c_test
[tPairs str_tPairs S] = officialTransferPairs(U, param);
clsnames = stringifyClasses(param.dataset);


U0 = U;
U.all = U0;
U.train = U0(:, trainPrototypes);
U.test = U0(:, testPrototypes);

numPrototypes.all = param.numPrototypes;
numPrototypes.train = param.numPrototypes(train_classes);
numPrototypes.test = param.numPrototypes(test_classes);

prototypes.train = trainPrototypes;
prototypes.test = testPrototypes;

for i=1:size(tPairs, 1)

    % Transfer direction : c1 ------->>> c2
    c1 = tPairs(i, 1);
    c2 = tPairs(i, 2);
    fprintf('\n\n\n================================ %s(%d) ---> %s(%d) ================================\n', clsnames{c1}, c1, clsnames{c2}, c2);

    trClass = find(train_classes == tPairs(i, 1));
    teClass = find(test_classes == tPairs(i, 2));
    scale_alpha = 1;
    [U_test numPrototypes_test] = officialTransfer(DS, W, U, numPrototypes, prototypes, trClass, teClass, scale_alpha);

    % Visualize
    % coord_idx = visualizePrototypes(U_new, param_new, coord_idx, inferred_idx);
    % pause;

    if length(inferred_idx) > 0
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
    end

    U_prev = U_new;
    param_prev = param_new;
    pause;
end