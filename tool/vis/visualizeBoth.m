function coord_idx = visualizeBoth(DS, W, U, param, coord_idx, special_prototypes)

% init plotting
f = figure;
hold on;
[classNames, colorList] = initPlotting(param);


% Coordinate Projection
if length(coord_idx) == 0
    [~, coord_idx] = sort(sum(bsxfun(@minus, U, mean(U, 2)).^2, 2), 'descend');
    coord_idx = coord_idx(1:3);
end



%%-------- Draw examples
WX = W*DS.D;
WX = WX(coord_idx, :);

for i=1:param.numClasses
    exmplIdx = DS.DL == i;
    plot3(WX(1, exmplIdx), WX(2, exmplIdx), WX(3, exmplIdx), '.', 'Color', colorList(i, :), 'MarkerSize', 5, 'DisplayName', classNames{i});
end



%%-------- Draw prototypes
class_labels = zeros(sum(param.numPrototypes), 1);
protoStartIdx = [0; cumsum(param.numPrototypes)];
for i=1:param.numClasses
    class_labels(protoStartIdx(i)+1:protoStartIdx(i+1)) = i;
end


if length(special_prototypes) > 0
    classNum = length(find(special_prototypes(1) > [0; cumsum(param.numPrototypes)]));
    inferred_class_name = [classNames{classNum} ' inferred']; 

    U_vis = U(coord_idx, :);
    U_vis_special = U_vis(:, special_prototypes);
    U_vis_remain = U_vis;
    U_vis_remain(:, special_prototypes) = [];

    class_labels_special = class_labels;
    class_labels_special(special_prototypes) = [];

    for i=1:param.numClasses
        protoIdx = class_labels_special == i;
        plot3(U_vis_remain(1, protoIdx), U_vis_remain(2, protoIdx), U_vis_remain(3, protoIdx), '.', 'Color', colorList(i, :), 'MarkerSize', 20, 'DisplayName', classNames{i});

        if classNum == i
            plot3(U_vis_special(1, :), U_vis_special(2, :), U_vis_special(3, :), '.', 'Color', colorList(i, :), 'MarkerSize', 50, 'DisplayName', inferred_class_name);
        end
    end

else
    U_vis = U(coord_idx, :);
    for i=1:param.numClasses
        protoIdx = class_labels == i;

        plot3(U_vis(1, protoIdx), U_vis(2, protoIdx), U_vis(3, protoIdx), '.', 'Color', colorList(i, :), 'MarkerSize', 20, 'DisplayName', classNames{i});
    end
end

axis off
legend('show', 'Location', 'SouthEast');

drawlines(U_vis, class_labels, colorList, param, 3);

hold off;
