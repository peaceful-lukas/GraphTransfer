% initial clustering
k = param.num_clusters;
[classProtos, param] = spectralClustering(DS, param, k);

% initialize U by pca of cluster prototypes
[~, pca_score, ~] = pca(classProtos');
tempU = pca_score(:, 1:param.lowDim)';



% initialize W with ridge regression
X = DS.D;
projection_lambda = 10000000;
J = arrayfun(@(p) repmat(tempU(:, p), 1, length(find(param.protoAssign == p)))*X(:, find(param.protoAssign == p))', 1:sum(param.numPrototypes), 'UniformOutput', false);
J = sum(cat(3, J{:}), 3);
W = J*pinv(X*X'+projection_lambda*eye(param.featureDim));


% re-initialize U by taking mean vectors of WX_c
proto_offset = [0; cumsum(param.numPrototypes)];
protoPool = {};
for classNum=1:param.numClasses
    protoPool{classNum} = [];

    for protoNum=proto_offset(classNum)+1:proto_offset(classNum)+param.numPrototypes(classNum)
        exampleIdx = find(param.protoAssign == protoNum);
        X = sum(W*DS.D(:, exampleIdx), 2);
        proto = WX/length(exampleIdx);
        protoPool{classNum} = [protoPool{classNum}, proto];
    end
end



% n = 0;
% highest_acc = 0.4;
% highest_W = W;
% highest_U = U;
% iter_condition = 1;

% while( n < param.maxAlter & iter_condition )
%     fprintf('\n============================= Iteration %d =============================\n', n+1);

%     prev_W = W;
%     prev_U = U;

%     W = learnW(DS, W, U, param);
%     U = learnU(DS, W, U, param);

%     [~, accuracy] = dispAccuracy(param.method, DS, W, U, param);

%     if exist('accuracy') && accuracy > highest_acc
%         saveResult(param.method, param.dataset, accuracy, {param, W, U, accuracy}, local_env);

%         highest_acc = accuracy;
%         highest_W = W;
%         highest_U = U;
%         fprintf('[splme] highest accuracy has been renewed. (acc = %.4f)\n', highest_acc);
%     end

%     iter_condition = sqrt((norm(W, 'fro') - norm(prev_W, 'fro'))^2 +  (norm(U, 'fro') - norm(prev_U, 'fro'))^2) > 0.000001;

%     n = n + 1;

%     if local_env && mod(n, 5) == 1
%         % coord_idx = visualizeBoth(DS, W, U, param, [], [], 'train');
%         coord_idx = visualizePrototypes(U, param, [], []);
%         drawnow;
%     end
% end
