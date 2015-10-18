function [U0 classProtos param] = initU(DS, param)



param.protoAssign = zeros(length(DS.DL), 1);
param.numPrototypes = zeros(param.numClasses, 1);
classProtos = [];

k = 10;
for classNum=1:param.numClasses
    exampleIdx = find(DS.DL == classNum);
    X_c = DS.D(:, exampleIdx);

    A_c = calculateAffinityMatrix(param.method, X_c);
    D_c = calculateDegreeMatrix(A_c);
    NL_c = calculateNormalizedLaplacian(D_c, A_c);

    % perform the eigen value decomposition
    [eigVectors,eigValues] = eig(NL_c);

    % select k largest eigen vectors
    nEigVec = eigVectors(:,end-(k-1):end);
    
    % construct the normalized matrix U from the obtained eigen vectors
    U_c = normr(nEigVec);

    % perform kmeans clustering on the matrix U
    [protoAssign, prototypes] = kmeans(U_c, k);

    classProtos = [classProtos prototypes'];

    protoAssign = protoAssign + sum(param.numPrototypes(1:classNum-1));
    param.protoAssign(exampleIdx) = protoAssign;
    param.numPrototypes(classNum) = k;


    % uniqueProtoIdx = unique(protoAssign);
    % for m=1:k
    %     similarExampleIdx = find(protoAssign == uniqueProtoIdx(m));
    %     sim_vec_m = nEigVec(similarExampleIdx, :)*prototypes(m, :)';
    %     [~, sim_sorted_idx]= sort(sim_vec_m, 'descend');

    %     imageIdx = exampleIdx(similarExampleIdx(sim_sorted_idx)); 

    %     fig = figure;
    %     set(fig, 'Position', [0, 700, 1300, 1000]);    
    %     for i=1:min(9, length(similarExampleIdx))
    %         subplot(3, 3, i);
    %         imagesc(DS.DI{imageIdx(i)});
    %         axis image;
    %         axis off;
    %     end

    %     pause;
    % end

    fprintf('class %d finished\n', classNum);
end

param.knnGraphs = constructKnnGraphs(classProtos, param);
U0 = rand(param.lowDim, sum(param.numPrototypes));




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
%     protoAssign(nullIdx) = sorted_distMat(:, 1) + sum(param.numPrototypes(1:c-1)) ;

%     param.protoAssign(class_idx) = protoAssign;
%     param.numPrototypes(c) = numel(unique(ta)) - numOutliers;
%     classProtos = [classProtos classProtos_c];

%     fprintf('class %d clustering finished ( # of clusters = %d )\n', c, param.numPrototypes(c));
% end


% % ------ should be connected graphs. MUST BE CHECKED. -----------
% param.knnGraphs = constructKnnGraphs(classProtos, param);

% % param.lowDim = sum(param.numPrototypes) - 1;
% [~, pca_score, ~] = pca(classProtos');
% U0 = pca_score(:, 1:param.lowDim)'; % approximate the original distributions of prototypes.
% % U0 = U0/norm(U0, 'fro');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Whether to divide by Frobenius norm or not??

