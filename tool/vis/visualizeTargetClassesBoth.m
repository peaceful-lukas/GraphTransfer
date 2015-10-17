function visualizeTargetClassesBoth(DS, W, U, param, coord_idx, targetClasses, datatype)

if nargin < 6
    datatype = 'train';
end



X = [];
Y = [];
if strcmp(datatype, 'train')
    X = DS.D;
    Y = DS.DL;
elseif strcmp(datatype, 'test')
    X = DS.T;
    Y = DS.TL;
end

%-------- init plotting
f = figure;
hold on;
[classNames, colorList] = initPlotting(param);


%-------- Select target classes
classNames = classNames(targetClasses);
colorList = colorList(targetClasses, :);
targetExamplesIdx = find(ismember(Y, targetClasses));
X = X(:, targetExamplesIdx);
Y = Y(targetExamplesIdx);



%%-------- Draw examples
WX = W*X;
WX = WX(coord_idx, :);

for i=1:length(targetClasses)
    exmplIdx = Y == i;
    plot3(WX(1, exmplIdx), WX(2, exmplIdx), WX(3, exmplIdx), '.', 'Color', colorList(i, :), 'MarkerSize', 5, 'DisplayName', classNames{i});
end


%%-------- Draw prototypes
class_labels = zeros(sum(param.numPrototypes), 1);
protoStartIdx = [0; cumsum(param.numPrototypes)];
for i=1:length(targetClasses)
    class_labels(protoStartIdx(i)+1:protoStartIdx(i+1)) = i;
end


U_vis = U(coord_idx, :);
for i=1:length(targetClasses)
    protoIdx = class_labels == i;

    plot3(U_vis(1, protoIdx), U_vis(2, protoIdx), U_vis(3, protoIdx), '.', 'Color', colorList(i, :), 'MarkerSize', 20, 'DisplayName', classNames{i});
end


axis off
legend('show', 'Location', 'SouthEast');



%------ Draw lines

for i=1:length(targetClasses)
    target = find(class_labels == i);
    points = U_vis(:, target);
    
    adjGraph = triu(param.knnGraphs{i});
    [p1 p2] = find(adjGraph);
    pairs = [p1 p2];

    for j=1:size(pairs, 1)
        p1 = points(:, pairs(j, 1));
        p2 = points(:, pairs(j, 2));
        p=[p2'; p1'];
        plot3(p(:,1), p(:,2), p(:,3), 'Color', colorList(i, :)); 
    end
end

hold off;



