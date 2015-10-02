function perAcc = perClassAccuracy(DS, W, U, param, datatype)

perAcc = zeros(param.numClasses, 1);

if strcmp(datatype, 'train')
    cumNumProto = cumsum(param.numPrototypes);
    [~, classified_raw] = max(DS.D'*W'*U, [], 2);
    classified = zeros(numel(classified_raw), 1);
    for c = 1:param.numClasses
        t = find(classified_raw <= cumNumProto(c));
        classified(t) = c;
        classified_raw(t) = Inf;
    end
    for i=1:param.numClasses
        class_idx = find(DS.DL == i);
        perAcc(i) = numel(find(classified(class_idx) == i))/numel(class_idx);
    end
elseif strcmp(datatype, 'test')
    cumNumProto = cumsum(param.numPrototypes);
    [~, classified_raw] = max(DS.T'*W'*U, [], 2);
    classified = zeros(numel(classified_raw), 1);
    for c = 1:param.numClasses
        t = find(classified_raw <= cumNumProto(c));
        classified(t) = c;
        classified_raw(t) = Inf;
    end
    for i=1:param.numClasses
        class_idx = find(DS.DL == i);
        perAcc(i) = numel(find(classified(class_idx) == i))/numel(class_idx);
    end
end