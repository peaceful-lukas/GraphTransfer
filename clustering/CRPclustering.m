function [classProtos, param] = CRPclustering(DS, param, visualize)

% % ddCRP clustering
param.protoAssign = zeros(length(DS.DL), 1);
param.numPrototypes = zeros(param.numClasses, 1);
classProtos = [];
for c = 1:param.numClasses
    X_c = DS.D(:, find(DS.DL == c));
    D = conDstMat(X_c);
    D = D./max(max(D));
    
    numData_c = size(X_c, 2);
    % alpha = numData_c * 0.01; % overfit
    alpha = numData_c * 0.1; % better
    a = mean(mean(D));
    [ta, ~] = ddcrp(D, 'lgstc', alpha, a);

    % prevent to catch outliers
    protoAssign = zeros(numData_c, 1);

    numOutliers = 0;
    clust = unique(ta);
    classProtos_c = [];
    for p=1:length(clust)
        
        if length(find(ta == p)) < 10 % outliers
        % if length(find(ta == p)) < 3 % outliers
            protoAssign(find(ta == p)) = -1;
            numOutliers = numOutliers + 1;
            fprintf('outliers of the class %d) # of outliers for prototype %d : %d\n', c, p, length(find(ta == p)));
        else
            protoAssign(find(ta == p)) = p - numOutliers + sum(param.numPrototypes(1:c-1));
            classProtos_c = [classProtos_c mean(X_c(:, find(ta == p)), 2)];
        end
    end

    classProtos = [classProtos classProtos_c];
    param.protoAssign(find(DS.DL == c)) = protoAssign;
    param.numPrototypes(c) = numel(unique(ta)) - numOutliers;

    fprintf('class %d clustering finished ( # of clusters = %d )\n', c, param.numPrototypes(c));


    if visualize
        uniqueProtoIdx = unique(protoAssign);
        for m=1:param.numPrototypes(c)
            similarExampleIdx = find(protoAssign == uniqueProtoIdx(m));
            sim_vec_m = X_c(:, similarExampleIdx)'*classProtos_c(:, m);
            [~, sim_sorted_idx]= sort(sim_vec_m, 'descend');

            % sim_sorted_idx = sim_sorted_idx(randperm(length(sim_vec_m)));
            exampleIdx = find(DS.DL == c);
            imageIdx = exampleIdx(similarExampleIdx(sim_sorted_idx)); 

            f = figure('Visible', 'off');
            set(f, 'Position', [0, 700, 1300, 1000]);    
            for i=1:min(9, length(similarExampleIdx))
                subplot(3, 3, i);
                imagesc(DS.DI{imageIdx(i)});
                axis image;
                axis off;
            end
            saveas(f, ['/Users/lukas/Desktop/meeting_fig/' num2str(c) '_' num2str(m) '.jpg']);
            % pause;
        end
    end
end

param.knnGraphs = constructKnnGraphs(classProtos, param);

