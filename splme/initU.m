function [U0 param] = initU(DS, param)

% % ddCRP clustering
param.protoAssign = zeros(length(DS.DL), 1);
param.numPrototypes = zeros(param.numClasses, 1);
classProtos = [];
for c = 1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));
    D = conDstMat(X_c);
    D = D./max(max(D));
    
    numData_c = size(X_c, 2);
    % alpha = numData_c * 0.01;
    alpha = numData_c * 0.1;
    a = mean(mean(D));
    [ta, ~] = ddcrp(D, 'lgstc', alpha, a);

    % prevent to catch outliers
    protoAssign = zeros(numData_c, 1);

    numOutliers = 0;
    clust = unique(ta);
    
    for p=1:length(clust)
        
        % if length(find(ta == p)) < 10 % outliers
        if length(find(ta == p)) < 3 % outliers
            protoAssign(find(ta == p)) = -1;
            numOutliers = numOutliers + 1;
            fprintf('outliers of the class %d) # of outliers for prototype %d : %d\n', c, p, length(find(ta == p)));
        else
            protoAssign(find(ta == p)) = p - numOutliers + sum(param.numPrototypes(1:c-1));
            classProtos = [classProtos mean(X_c(:, find(ta == p)), 2)];
        end
    end
    
    param.protoAssign(find(DS.DL == c)) = protoAssign;
    param.numPrototypes(c) = numel(unique(ta)) - numOutliers;

    fprintf('class %d clustering finished ( # of clusters = %d )\n', c, param.numPrototypes(c));
end

% ------ should be connected graphs. MUST BE CHECKED. -----------
[param.sTriplets knnGraphs] = generateStructurePreservingTriplets(classProtos, param);
param.knnGraphs = knnGraphs;

% param.lowDim = sum(param.numPrototypes) - 1;
[~, pca_score, ~] = pca(classProtos');
U0 = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
U0 = normc(U0);
% max_norm = max(arrayfun(@(n) norm(U0(:, n)), 1:size(U0, 2)));
% U0 = U0/max_norm;

