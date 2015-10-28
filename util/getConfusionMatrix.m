function [C predicted] = getConfusionMatrix(DS, W, U, param, methodType)

if strcmp(methodType, 'sim')
    cumNumProto = cumsum(param.numPrototypes);
    [~, predicted_raw] = max(DS.T'*W'*U, [], 2);
    predicted = zeros(numel(predicted_raw), 1);
    for c = 1:param.numClasses
        t = find(predicted_raw <= cumNumProto(c));
        predicted(t) = c;
        predicted_raw(t) = Inf;
    end
    test_acc = numel(find(DS.TL == predicted))/numel(DS.TL);


    C = confusionmat(DS.TL, predicted);


elseif strcmp(methodType, 'dist')

    cumNumProto = cumsum(param.numPrototypes);
    D = pdist2((W*DS.T)', U');
    [~, predicted_raw] = min(D, [], 2);
    predicted = zeros(numel(predicted_raw), 1);
    for c = 1:param.numClasses
        t = find(predicted_raw <= cumNumProto(c));
        predicted(t) = c;
        predicted_raw(t) = Inf;
    end
    test_acc = numel(find(DS.TL == predicted))/numel(DS.TL);


    C = confusionmat(DS.TL, predicted);

end