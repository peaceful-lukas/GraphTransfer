function [train_acc test_acc] = dispAccuracy(method, DS, W, U, param)

    if strcmp(method, 'splme') || strcmp(method, 'splme_new') || strcmp(method, 'splme_sim') || strcmp(method, 'splme_sim_const')

        cumNumProto = cumsum(param.numPrototypes);
        [~, classified_raw] = max(DS.D'*W'*U, [], 2);
        classified = zeros(numel(classified_raw), 1);
        for c = 1:param.numClasses
            t = find(classified_raw <= cumNumProto(c));
            classified(t) = c;
            classified_raw(t) = Inf;
        end
        train_acc = numel(find(DS.DL == classified))/numel(DS.DL);
        fprintf('TRAIN accuracy : %.4f\n', train_acc);


        cumNumProto = cumsum(param.numPrototypes);
        [~, classified_raw] = max(DS.T'*W'*U, [], 2);
        classified = zeros(numel(classified_raw), 1);
        for c = 1:param.numClasses
            t = find(classified_raw <= cumNumProto(c));
            classified(t) = c;
            classified_raw(t) = Inf;
        end
        test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
        fprintf('TEST accuracy :  %.4f\n', test_acc);
    
        
        
    elseif strcmp(method, 'splme_dist') || strcmp(method, 'splme_dist_const') || strcmp(method, 'spcl') || strcmp(method, 'splme_cvpr')
        cumNumProto = cumsum(param.numPrototypes);
        D = arrayfun(@(p) sum((W*DS.D - repmat(U(:, p), 1, length(DS.DL))).^2, 1), 1:sum(param.numPrototypes), 'UniformOutput', false);
        D = cat(1, D{:});
        [~, classified_raw] = min(D, [], 1);
        classified = zeros(numel(classified_raw), 1);
        for c = 1:param.numClasses
            t = find(classified_raw <= cumNumProto(c));
            classified(t) = c;
            classified_raw(t) = Inf;
        end
        train_acc = numel(find(DS.DL == classified))/numel(DS.DL);
        fprintf('TRAIN accuracy : %.4f\n', train_acc);


        cumNumProto = cumsum(param.numPrototypes);
        D = arrayfun(@(p) sum((W*DS.T - repmat(U(:, p), 1, length(DS.TL))).^2, 1), 1:sum(param.numPrototypes), 'UniformOutput', false);
        D = cat(1, D{:});
        [~, classified_raw] = min(D, [], 1);
        classified = zeros(numel(classified_raw), 1);
        for c = 1:param.numClasses
            t = find(classified_raw <= cumNumProto(c));
            classified(t) = c;
            classified_raw(t) = Inf;
        end
        test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
        fprintf('TEST accuracy :  %.4f\n', test_acc);
        
        
    elseif strcmp(method, 'blme_dist')
        D = arrayfun(@(c) sum((W*DS.D - repmat(U(:, c), 1, length(DS.DL))).^2, 1), 1:param.numClasses, 'UniformOutput', false);
        D = cat(1, D{:});
        [~, classified] = min(D, [], 1);
        train_acc = numel(find(DS.DL == classified'))/numel(DS.DL);
        fprintf('TRAIN accuracy : %.4f\n', train_acc);

        D = arrayfun(@(c) sum((W*DS.T - repmat(U(:, c), 1, length(DS.TL))).^2, 1), 1:param.numClasses, 'UniformOutput', false);
        D = cat(1, D{:});
        [~, classified] = min(D, [], 1);
        test_acc = numel(find(DS.TL == classified'))/numel(DS.TL);
        fprintf('TEST accuracy :  %.4f\n', test_acc);



    elseif strcmp(method, 'blme_sim')
        [~, classified] = max(DS.D'*W'*U, [], 2);
        train_acc = numel(find(DS.DL == classified))/numel(DS.DL);
        fprintf('TRAIN accuracy : %.4f\n', train_acc);

        [~, classified] = max(DS.T'*W'*U, [], 2);
        test_acc = numel(find(DS.TL == classified))/numel(DS.TL);
        fprintf('TEST accuracy :  %.4f\n', test_acc);
    end

end