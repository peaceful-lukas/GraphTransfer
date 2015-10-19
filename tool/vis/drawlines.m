function drawlines(U, U_low, labels, colorList, param, dim)
% function drawlines(U, U_low, labels, colorList, param, dim, i)



for i=1:param.numClasses
    target = find(labels == i);
    points = U_low(:, target);
    % 
    % POINTS = U(:, target);
    
    adjGraph = triu(param.knnGraphs{i});
    [p1 p2] = find(adjGraph);
    pairs = [p1 p2];

 
    for j=1:size(pairs, 1)
        drawline(points(:, pairs(j, 1)), points(:, pairs(j, 2)), colorList(i, :), dim);
        % 
        % markDistance(POINTS(:, pairs(j, 1)), POINTS(:, pairs(j, 2)), points(:, pairs(j, 1)), points(:, pairs(j, 2)));
    end

    % print out neighbors
    % keyboard;
    % knnGraph = param.knnGraphs{i};
    % for j=1:size(knnGraph, 1)
    %     neighbors = find(knnGraph(j, :));
    %     markNeighbors(points(:, j), neighbors, j);
    % end
end


function [] = drawline(p1, p2, color, dim)

if dim == 2
    p=[p2'; p1'];
    plot(p(:,1), p(:,2), 'Color', color);
elseif dim == 3
    p=[p2'; p1'];
    plot3(p(:,1), p(:,2), p(:,3), 'Color', color);
end


% 
function markDistance(P1, P2, p1, p2)

dist = sqrt(sum((P1-P2).^2, 1));

m = (p1+p2)/2;
text(m(1), m(2), m(3), sprintf('%.4f', dist));



function markNeighbors(point, neighbors, nodeNum)
% keyboard;
neighbors_str = sprintf('(%d) ', nodeNum);
for i=1:length(neighbors)
    neighbors_str = [neighbors_str sprintf('%d,', neighbors(i))];
end

text(point(1), point(2), point(3), neighbors_str);