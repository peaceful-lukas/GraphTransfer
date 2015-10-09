function [U0 param] = initU(DS, param)

% dd-CRP clustering
param.protoAssign = zeros(length(DS.DL), 1);
param.numPrototypes = zeros(param.numClasses, 1);
classProtos = [];

for c = 1:param.numClasses
    class_idx = find(DS.DL == c);
    X_c = DS.D(:, class_idx);
    D = conDstMat(X_c);
    D = D./max(max(D));
    
    numData_c = size(X_c, 2);
    alpha = numData_c * 0.1; % scale parameter: larger is greater scale
    a = 0;
    [ta, ~] = ddcrp(D, 'lgstc', alpha, a);

    % prevent to catch outliers
    protoAssign = zeros(numData_c, 1);
    classProtos_c = [];

    numOutliers = 0;
    clust = unique(ta);
    
    for p=1:length(clust)
        protoIdx = find(ta == p);
        
        if length(protoIdx) < 8 % outliers
            protoAssign(protoIdx) = -1;
            numOutliers = numOutliers + 1;
            % fprintf('outliers of the class %d) # of outliers for prototype %d : %d\n', c, p, length(protoIdx));
        else
            protoNum = p - numOutliers + sum(param.numPrototypes(1:c-1));
            protoAssign(protoIdx) = protoNum;

            classProtos_c = [classProtos_c mean(X_c(:, protoIdx), 2)];
        end
    end
    
    % re-assign to the nearest prototype
    nullIdx = find(protoAssign == -1);

    distMat = zeros(length(nullIdx), size(classProtos_c, 2));
    for i=1:length(nullIdx)
        distMat(i, :) = sum(bsxfun(@minus, classProtos_c, X_c(:, nullIdx(i))).^2, 1);
    end
    [~, sorted_distMat] = sort(distMat, 2, 'ascend');
    protoAssign(nullIdx) = sorted_distMat(:, 1) + sum(param.numPrototypes(1:c-1)) ;

    param.protoAssign(class_idx) = protoAssign;
    param.numPrototypes(c) = numel(unique(ta)) - numOutliers;
    classProtos = [classProtos classProtos_c];

    fprintf('class %d clustering finished ( # of clusters = %d )\n', c, param.numPrototypes(c));
end


% ------ should be connected graphs. MUST BE CHECKED. -----------
param.knnGraphs = constructKnnGraphs(classProtos, param);

% param.lowDim = sum(param.numPrototypes) - 1;
[~, pca_score, ~] = pca(classProtos');
U0 = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U0 = U0/norm(U0, 'fro');
% max_norm = max(arrayfun(@(n) norm(U0(:, n)), 1:size(U0, 2)));
% U0 = U0/max_norm;

