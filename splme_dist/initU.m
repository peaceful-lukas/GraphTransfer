function [U0 classProtos param] = initU(DS, param)


param.protoAssign = zeros(length(DS.DL), 1);
param.numPrototypes = zeros(param.numClasses, 1);
classProtos = [];

% Spectral Clustering with k (= lowDim)
k = param.lowDim;

% graph Laplacian
X = DS.D;
A = calculateAffinityMatrix(param.method, X);
D = calculateDegreeMatrix(A);
NL = calculateNormalizedLaplacian(D, A);

% perform the eigen value decomposition & select k largest eigen vectors
[eigVectors, eigValues] = eig(NL);
eigX = eigVectors(:,end-(k-1):end);
neigX = eigX;%normr(eigX);

% perform kmeans clustering on the matrix U
[protoAssign, prototypes] = kmeans(neigX, k);
prototypes = prototypes';

% prototype class assignment
voting_k = 5; % knn-voting constant
prototypeClasses = zeros(k, 1);
for p=1:k
    proto = prototypes(:, p);
    dist_vec = sum(bsxfun(@minus, neigX', proto).^2);
    [~, nearest_idx] = sort(dist_vec, 'ascend');
    nearest_classes = DS.DL(nearest_idx(1:voting_k));
    prototypeClasses(p) = mode(nearest_classes);
end

U0 = [];
for classNum=1:param.numClasses
     protoIdx = find(prototypeClasses == classNum);
     U0 = [U0 prototypes(:, protoIdx)];
     param.numPrototypes(classNum) = length(protoIdx);
end

param.numPrototypes



% classNum = 1;
for classNum=1:param.numClasses
    exampleIdx = find(DS.DL == classNum);
    X_c = DS.D(:, exampleIdx);

    A_c = calculateAffinityMatrix(param.method, X_c, param);
    D_c = calculateDegreeMatrix(A_c);
    NL_c = calculateNormalizedLaplacian(D_c, A_c);

    % perform the eigen value decomposition
    [eigVectors,eigValues] = eig(NL_c);

    % select k largest eigen vectors
    nEigVec = eigVectors(:,end-(k-1):end);
    
    % construct the normalized matrix U from the obtained eigen vectors
    U_c = zeros(size(nEigVec));
    for i=1:size(nEigVec,1)
        n = sqrt(sum(nEigVec(i, :).^2));    
        U_c(i, :) = nEigVec(i, :) ./ n; 
    end

    % perform kmeans clustering on the matrix U
    [protoAssign, prototypes] = kmeans(U_c, k);

    classProtos = [classProtos prototypes'];

    protoAssign = protoAssign + sum(param.numPrototypes(1:classNum-1));
    param.protoAssign(exampleIdx) = protoAssign;
    param.numPrototypes(classNum) = k;

    fprintf('class %d finished\n', classNum);
end








% % dd-CRP clustering
% param.protoAssign = zeros(length(DS.DL), 1);
% param.numPrototypes = zeros(param.numClasses, 1);
% classProtos = [];

% for c = 1:param.numClasses
%     class_idx = find(DS.DL == c);
%     X_c = DS.D(:, class_idx);
%     normX_c = normc(X_c);
%     S = normX_c'*normX_c;
    
%     numData_c = size(X_c, 2);
%     % 0.05 / 0.1
%     alpha = numData_c * 0.1; % scale parameter: larger is greater scale 

%     a = 0;
%     [ta, ~] = ddcrp(S, 'lgstc', alpha, a);
%     % [ta, ~] = sim_crp(S, 'lgstc', alpha, a);

%     % prevent to catch outliers
%     protoAssign = zeros(numData_c, 1);
%     classProtos_c = [];

%     numOutliers = 0;
%     clust = unique(ta);
    
%     for p=1:length(clust)
%         protoIdx = find(ta == p);
        
%         if length(protoIdx) < 8 % outliers
%             protoAssign(protoIdx) = -1;
%             numOutliers = numOutliers + 1;
%             % fprintf('outliers of the class %d) # of outliers for prototype %d : %d\n', c, p, length(protoIdx));
%         else
%             protoNum = p - numOutliers + sum(param.numPrototypes(1:c-1));
%             protoAssign(protoIdx) = protoNum;

%             classProtos_c = [classProtos_c mean(X_c(:, protoIdx), 2)];
%         end
%         fprintf('example assigned of class %d) # of examples for prototype %d : %d\n', c, p, length(protoIdx));
%     end
    
%     % re-assign to the nearest prototype
%     nullIdx = find(protoAssign == -1);

%     distMat = zeros(length(nullIdx), size(classProtos_c, 2));
%     for i=1:length(nullIdx)
%         distMat(i, :) = sum(bsxfun(@minus, classProtos_c, X_c(:, nullIdx(i))).^2, 1);
%     end
%     [~, sorted_distMat] = sort(distMat, 2, 'ascend');
%     protoAssign(nullIdx) = sorted_distMat(:, 1) + sum(param.numPrototypes(1:c-1));

%     param.protoAssign(class_idx) = protoAssign;
%     param.numPrototypes(c) = numel(unique(ta)) - numOutliers;
%     classProtos = [classProtos classProtos_c];

%     fprintf('class %d clustering finished ( # of clusters = %d )\n', c, param.numPrototypes(c));
% end


% % % ------ should be connected graphs. MUST BE CHECKED. -----------
% param.knnGraphs = constructKnnGraphs(classProtos, param);

% [~, pca_score, ~] = pca(classProtos');
% U0 = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.

