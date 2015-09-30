function drawlines(U_low, labels, colorList, param, dim)



for i=1:param.numClasses
    target = find(labels == i);
    points = U_low(:, target);
    
    adjGraph = triu(param.knnGraphs{i});
    [p1 p2] = find(adjGraph);
    pairs = [p1 p2];

    for j=1:size(pairs, 1)
        drawline(points(:, pairs(j, 1)), points(:, pairs(j, 2)), colorList(i, :), dim);
    end
end


function [] = drawline(p1, p2, color, dim)

if dim == 2
    p=[p2'; p1'];
    plot(p(:,1), p(:,2), 'Color', color);
elseif dim == 3
    p=[p2'; p1'];
    plot3(p(:,1), p(:,2), p(:,3), 'Color', color);
end