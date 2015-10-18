function [classProtos, param] = kmeansClustering(DS, param, k, visualize)


if nargin < 4
    visualize = false;
end

param.protoAssign = zeros(length(DS.DL), 1);
param.numPrototypes = zeros(param.numClasses, 1);
classProtos = [];

for classNum=1:param.numClasses
    exampleIdx = find(DS.DL == classNum);
    X_c = DS.D(:, exampleIdx);

    [protoAssign, prototypes] = kmeans(X_c', k);

    classProtos = [classProtos prototypes'];

    protoAssign = protoAssign + sum(param.numPrototypes(1:classNum-1));
    param.protoAssign(exampleIdx) = protoAssign;
    param.numPrototypes(classNum) = k;

    fprintf('class %d finished\n', classNum);


    if visualize
        uniqueProtoIdx = unique(protoAssign);
        for m=1:k
            similarExampleIdx = find(protoAssign == uniqueProtoIdx(m));
            sim_vec_m = X_c(:, similarExampleIdx)'*prototypes(m, :)';
            [~, sim_sorted_idx]= sort(sim_vec_m, 'descend');

            imageIdx = exampleIdx(similarExampleIdx(sim_sorted_idx)); 

            fig = figure;
            set(fig, 'Position', [0, 700, 1300, 1000]);    
            for i=1:min(9, length(similarExampleIdx))
                subplot(3, 3, i);
                imagesc(DS.DI{imageIdx(i)});
                axis image;
                axis off;
            end

            pause;
        end
    end
end

param.knnGraphs = constructKnnGraphs(classProtos, param);

