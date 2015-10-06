function coord_idx = visualizeExamples(DS, W, U, param, coord_idx)

% init plotting
f = figure;
hold on;
[classNames, colorList] = initPlotting(param);

% Coordinate Projection
if length(coord_idx) == 0
    [~, coord_idx] = sort(sum(bsxfun(@minus, U, mean(U, 2)).^2, 2), 'descend');
    coord_idx = coord_idx(1:3);
end

WX = W*DS.D;
WX = WX(coord_idx, :);

for i=1:param.numClasses
    exmplIdx = DS.DL == i;
    plot3(WX(1, exmplIdx), WX(2, exmplIdx), WX(3, exmplIdx), '.', 'Color', colorList(i, :), 'MarkerSize', 5, 'DisplayName', classNames{i});
end

axis off
legend('show', 'Location', 'SouthEast');

hold off;
