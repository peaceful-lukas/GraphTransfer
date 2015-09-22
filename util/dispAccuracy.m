function [train_acc test_acc] = dispAccuracy(method, DS, W, U, param)

    if strcmp(method, 'splme')

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

    end

end