% function [W U param] = splme_dist(DS, param, local_env)

% init U
[U classProtos param] = initU(DS, param);

% init W
W = rand(param.lowDim, param.featureDim);
% X = DS.D;
% projection_lambda = 1000000;
% J = arrayfun(@(p) repmat(U(:, p), 1, length(find(param.protoAssign == p)))*X(:, find(param.protoAssign == p))', 1:sum(param.numPrototypes), 'UniformOutput', false);
% J = sum(cat(3, J{:}), 3);
% W = J*pinv(X*X'+projection_lambda*eye(param.featureDim));

% if local_env
%     visualizeBoth(DS, W, U, param, [], [], 'test');
%     drawnow;
% end

[~, accuracy] = dispAccuracy(param.method, DS, W, U, param);



n = 0;
highest_acc = 0.5;
highest_W = W;
highest_U = U;
iter_condition = 1;
coord_idx = [];

while( n < param.maxAlter & iter_condition )
    fprintf('\n============================= Iteration %d =============================\n', n+1);

    prev_W = norm(W, 'fro');
    prev_U = norm(U, 'fro');

    W = learnW(DS, W, U, param);
    U = learnU(DS, W, U, param);

    [~, accuracy] = dispAccuracy(param.method, DS, W, U, param);

    if accuracy > highest_acc
        % perClassScores = perClassScore(DS, W, U, param);
        % saveResult(param.method, param.dataset, accuracy, {param, W, U, perClassScores, accuracy}, local_env);
        saveResult(param.method, param.dataset, accuracy, {param, W, U, accuracy}, local_env);

        highest_acc = accuracy;
        highest_W = W;
        highest_U = U;
        fprintf('[splme] highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.000001;

    n = n + 1;


    if local_env && mod(n, 5) == 1
        coord_idx = visualizeBoth(DS, W, U, param, [], [], 'test');
        drawnow;
        % pause;
    end
end

W = highest_W;
U = highest_U;

% coord_idx = visualizePrototypes(U, param, [], []);
% coord_idx = visualizeBoth(DS, W, U, param, [], []);




