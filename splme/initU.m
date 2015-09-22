function [U0 param] = initU(DS, param)

% % ddCRP clustering
protoAssign = zeros(length(DS.DL), 1);
numPrototypes = zeros(param.numClasses, 1);
classProtos = [];
for c = 1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));
    D = conDstMat(X_c);
    D = D./max(max(D));
    
    numData_c = size(X_c, 2);
    alpha = numData_c * 0.01;
    a = mean(mean(D));
    [ta, ~] = ddcrp(D, 'lgstc', alpha, a);
    numPrototypes(c) = numel(unique(ta));
    protoAssign(find(DS.DL == c)) = ta + sum(numPrototypes(1:c-1));

    % centroids of each cluster by examining ta
    for p = 1:numel(unique(ta))
        classProtos = [classProtos mean(X_c(:, find(ta == p)), 2)];
    end

    fprintf('class %d clustering finished ( # of clusters = %d )\n', c, numPrototypes(c));
end

% ------ should be connected graphs. MUST BE CHECKED. -----------
param.protoAssign = protoAssign;
param.numPrototypes = numPrototypes;
[param.sTriplets knnGraphs] = generateStructurePreservingTriplets(classProtos, param);
param.knnGraphs = knnGraphs;


[~, pca_score, ~] = pca(classProtos');
U0 = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U0 = normc(U0);