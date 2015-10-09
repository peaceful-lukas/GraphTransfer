function [C predicted] = getConfusionMatrix(DS, W, U, param)

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