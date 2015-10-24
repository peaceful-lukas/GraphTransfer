function [classProtos, param] = spectralClustering(DS, param, k, visualize)

if nargin < 4
    visualize = false;
end

param.protoAssign = zeros(length(DS.DL), 1);
param.numPrototypes = zeros(param.numClasses, 1);
classProtos = [];

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
    [protoAssign, eigPrototypes] = kmeans(U_c, k);

    % find the nearest examples from each prototype to set pseudo-prototypes
    prototypes = [];
    for p=1:k
        similarExampleIdx = find(protoAssign == p);
        sim_vec_m = nEigVec(similarExampleIdx, :)*eigPrototypes(p, :)';
        [~, protoIdx]= max(sim_vec_m);
        
        prototypes = [prototypes X_c(:, similarExampleIdx(protoIdx))];
    end

    classProtos = [classProtos prototypes];

    protoAssign = protoAssign + sum(param.numPrototypes(1:classNum-1));
    param.protoAssign(exampleIdx) = protoAssign;
    param.numPrototypes(classNum) = k;

    fprintf('class %d finished\n', classNum);

    if visualize
        uniqueProtoIdx = unique(protoAssign);
        for m=1:k
            similarExampleIdx = find(protoAssign == uniqueProtoIdx(m));
            sim_vec_m = nEigVec(similarExampleIdx, :)*eigPrototypes(m, :)';
            [~, sim_sorted_idx]= sort(sim_vec_m, 'descend');

            % sim_sorted_idx = sim_sorted_idx(randperm(length(sim_vec_m)));

            imageIdx = exampleIdx(similarExampleIdx(sim_sorted_idx)); 

            f = figure('Visible', 'off');
            set(f, 'Position', [0, 700, 1300, 1000]);    
            for i=1:min(9, length(similarExampleIdx))
                subplot(3, 3, i);
                imagesc(DS.DI{imageIdx(i)});
                axis image;
                axis off;
            end
            saveas(f, ['/Users/lukas/Desktop/meeting_fig/awa_pca500_' num2str(classNum) '_' num2str(m) '.jpg']);
            % pause;
        end
    end
end

param.nnGraphs = constructNNGraphs(classProtos, param);

