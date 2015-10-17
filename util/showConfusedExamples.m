function showConfusedExamples(DS, W, U, param, gt, pl)

cumNumProto = cumsum(param.numPrototypes);
[~, predicted_raw] = max(DS.T'*W'*U, [], 2);
predicted = zeros(numel(predicted_raw), 1);
for c = 1:param.numClasses
    t = find(predicted_raw <= cumNumProto(c));
    predicted(t) = c;
    predicted_raw(t) = Inf;
end
% test_acc = numel(find(DS.TL == predicted))/numel(DS.TL);

gtIdx = find(DS.TL == gt);
plIdx = find(predicted == pl);

target_idx = intersect(gtIdx, plIdx);
% keyboard;
f = figure;
set(f, 'Position', [100, 1300, 1000, 1000]);
hold on;

for i=1:min(length(target_idx), 15)
    subplot(3, 5, i);
    img = DS.TI{target_idx(i)};
    imagesc(img);
    axis off;
    axis image;
end

hold off;