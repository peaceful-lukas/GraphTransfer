function nearestExamples(DS, W, U, param, classNum, dataType)

if nargin < 6
    dataType = 'train';
end

X = [];
Y = [];
I = [];
if strcmp(dataType, 'train')
    X = DS.D;
    Y = DS.DL;
    I = DS.DI;
else strcmp(dataType, 'test')
    X = DS.T;
    Y = DS.TL;
    I = DS.TI;
end


cumProtos = [0; cumsum(param.numPrototypes)];
classProtoIdx = cumProtos(classNum)+1:cumProtos(classNum+1);

for p=classProtoIdx
    protoNum = p;
    class_example_idx = find(Y == classNum);

    WX_c = W*X(:, class_example_idx);
    u_c = U(:, protoNum);

    dist_vec = sum(bsxfun(@minus, WX_c, u_c).^2, 1);
    [dist_sorted, sorted_idx] = sort(dist_vec, 'ascend');

    I_c = {I{class_example_idx}};
    I_c_proto = {I_c{sorted_idx(1:9)}};

    f = figure;
    set(f, 'Position', [100, 1300, 1000, 1000]);
    hold on;

    for n=1:9
        subplot(3, 3, n);
        img = I_c_proto{n};
        imagesc(img);
        axis off;
        axis image;
    end

    pause;
end