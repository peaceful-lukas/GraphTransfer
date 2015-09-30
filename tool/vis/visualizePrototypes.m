function coord_idx = visualizePrototypes(U, param, coord_idx, special_prototypes)

[classNames, colorList] = initPlotting(param);

class_labels = zeros(sum(param.numPrototypes), 1);
protoStartIdx = [0; cumsum(param.numPrototypes)];
for i=1:param.numClasses
    class_labels(protoStartIdx(i)+1:protoStartIdx(i+1)) = i;
end

% Coordinate Projection
if length(coord_idx) == 0
    [~, coord_idx] = sort(sum(bsxfun(@minus, U, mean(U, 2)).^2, 2), 'descend');
end

if length(special_prototypes) > 0
    classNum = length(find(special_prototypes(1) > [0; cumsum(param.numPrototypes)]));
    inferred_class_name = [classNames{classNum} ' inferred']; 

    U_vis = U(coord_idx(1:3), :);
    U_vis_special = U_vis(:, special_prototypes);
    U_vis_remain = U_vis;
    U_vis_remain(:, special_prototypes) = [];

    class_labels_special = class_labels;
    class_labels_special(special_prototypes) = [];

    f = figure;
    hold on;
    for i=1:param.numClasses
        protoIdx = class_labels_special == i;
        plot3(U_vis_remain(1, protoIdx), U_vis_remain(2, protoIdx), U_vis_remain(3, protoIdx), '.', 'Color', colorList(i, :), 'MarkerSize', 20, 'DisplayName', classNames{i});

        if classNum == i
            plot3(U_vis_special(1, :), U_vis_special(2, :), U_vis_special(3, :), '.', 'Color', colorList(i, :), 'MarkerSize', 50, 'DisplayName', inferred_class_name);
        end
    end
    
    axis off
    legend('show', 'Location', 'Best');

    drawlines(U_vis, class_labels, colorList, param, 3);

else
    U_vis = U(coord_idx(1:3), :);

    f = figure;
    hold on;
    for i=1:param.numClasses
        protoIdx = class_labels == i;

        plot3(U_vis(1, protoIdx), U_vis(2, protoIdx), U_vis(3, protoIdx), '.', 'Color', colorList(i, :), 'MarkerSize', 20, 'DisplayName', classNames{i});
    end

    axis off
    legend('show', 'Location', 'Best');

    drawlines(U_vis, class_labels, colorList, param, 3);

    hold off;
end

% mouseEventCallback(f);



function [classNames, colorList] = initPlotting(param)

if strcmp(param.dataset, 'pascal3d_pascal')
    classNames = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'dtable', 'motorbike', 'sofa', 'train', 'tv'};

elseif strcmp(param.dataset, 'awa')
    classNames = {'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', ...
                'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', ...
                'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', ...
                'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', ...
                'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', ...
                'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', ...
                'wolf', 'chihuahua', 'rat', 'weasel', 'otter', ...
                'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', ...
                'pig', 'lion', 'mouse', 'polar+bear', 'collie',  ...
                'walrus', 'raccoon', 'cow', 'dolphin'};
end

colorList = distinguishable_colors(param.numClasses);




function mouseEventCallback(f)

set (gcf, 'WindowButtonMotionFcn', @mouseMove);

function mouseMove(object, eventdata)

C = get(gca, 'CurrentPoint');
title(gca, ['(X,Y) = (', num2str(C(1,1)), ', ',num2str(C(1,2)), ')']);