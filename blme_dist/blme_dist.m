% function [W U] = blme_dist(DS, param)

% initialize prototypes by PCA with mean values of datasets for each class
U_feature = zeros(param.featureDim, param.numClasses);
for n=1:param.numClasses
    U_feature(:, n) = mean(DS.D(:, find(DS.DL == n)), 2);
end
[~, pca_score, ~] = pca(U_feature');
pca_score = [pca_score ones(param.lowDim, 1)];
U = pca_score(:, 1:param.lowDim)';


X = DS.D;
projection_lambda = 1000000;
J = arrayfun(@(p) repmat(U(:, p), 1, length(find(DS.DL == p)))*X(:, find(DS.DL == p))', 1:size(U, 2), 'UniformOutput', false);
J = sum(cat(3, J{:}), 3);
W = J*pinv(X*X'+projection_lambda*eye(param.featureDim));

param.numPrototypes = ones(size(U, 2), 1);
visualizeBoth(DS, W, U, param, [], [], 'test');
[~, accuracy] = dispAccuracy(param.method, DS, W, U, param);
drawnow;

n = 0;
highest_acc = 0;
while( n < param.maxAlter )
    fprintf('\n============================= Iteration %d =============================\n', n+1);
    W = learnW(DS, W, U, param);
    U = learnU(DS, W, U, param);

    [~, accuracy] = dispAccuracy(method, DS, W, U, param);
    if accuracy > highest_acc
        saveResult(method, param.dataset, accuracy, {param, W, U, accuracy}, local_env);
        highest_acc = accuracy;
        fprintf('highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
    end

    n = n + 1;
end
