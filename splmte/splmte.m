% initial clustering
k = param.num_clusters;
[classProtos, param] = spectralClustering(DS, param, k);

% initialize U by pca of cluster prototypes
[~, pca_score, ~] = pca(classProtos');
U = pca_score(:, 1:param.lowDim)';


% initialize W with ridge regression
% tic
X = DS.D;
projection_lambda = 10000000;
J = arrayfun(@(p) repmat(U(:, p), 1, length(find(param.protoAssign == p)))*X(:, find(param.protoAssign == p))', 1:sum(param.numPrototypes), 'UniformOutput', false);
J = sum(cat(3, J{:}), 3);
W = J*pinv(X*X'+projection_lambda*eye(param.featureDim));

% W = randn(param.lowDim, param.featureDim);

% re-initialize U by taking mean vectors of WX_c
U = [];
for i=1:sum(param.numPrototypes)
    exampleIdx = find(param.protoAssign == i);
    WX = sum(W*DS.D(:, exampleIdx), 2);
    u = WX/length(exampleIdx);
    U = [U u];
end



% fprintf('Before Training...\n');
% [~, accuracy] = dispAccuracy(param.method, DS, W, U, param);

% visualize data distribution / class prototypes
% if local_env
%     visualizeBoth(DS, W, U, param, [], [], 'train');
%     visualizePrototypes(U, param, [], []);
%     drawnow;
% end

W = W0;
U = U0;


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
        saveResult(param.method, param.dataset, accuracy, {param, W, U, accuracy}, local_env);

        highest_acc = accuracy;
        highest_W = W;
        highest_U = U;
        fprintf('[splme] highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    iter_condition = sqrt((norm(W, 'fro') - prev_W)^2 +  (norm(U, 'fro') - prev_U)^2) > 0.000001;
    n = n + 1;

    % if local_env && mod(n, 5) == 1
    %     coord_idx = visualizeBoth(DS, W, U, param, [], [], 'test');
    %     drawnow;
    %     % pause;
    % end
end

W = highest_W;
U = highest_U;

% coord_idx = visualizePrototypes(U, param, [], []);
% coord_idx = visualizeBoth(DS, W, U, param, [], []);




