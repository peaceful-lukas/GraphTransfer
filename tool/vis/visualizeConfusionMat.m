function visualizeConfusionMat(DS, C, pr_labels, classNum, opts)
% opts - 

img_list = {};
gt_labels = DS.TL;


if nargin < 5
    opts.type = 'predicted';
    opts.numRows = 5;
    opts.numCols = 5;
    opts.title = 'Before Transfer';
end

visType = opts.type;
numRows = opts.numRows;
numCols = opts.numCols;
figTitle = opts.title;

if strcmp(visType, 'predicted')
    conf_vec = C(:, classNum);
    conf_vec(classNum) = 0; % remove correct prediction
    conf_classes = find(conf_vec);

    conf_idx = intersect(find(pr_labels == classNum), find(ismember(gt_labels, conf_classes)));
    img_list = cell(sum(conf_vec), 1); % sum(conf_vec) should equal to length(conf_idx)
    img_list = DS.TI(conf_idx);

    numConfusion = length(conf_idx);
    numIters = ceil(numConfusion/(numRows*numCols));

    fig = figure;
    hold on;
    set(fig, 'Position', [0, 700, 1300, 1000], 'name', figTitle);

    for iter=1:numIters
        fprintf('iter %d/%d\n', iter, numIters);
        clf('reset');
        for i=(iter-1)*numRows*numCols+1:iter*numRows*numCols
            if i > length(conf_idx), break; end
            
            plotPos = mod(i, numRows*numCols);
            if plotPos == 0, plotPos = numRows*numCols; end

            subplot(numRows, numCols, plotPos);
            imagesc(img_list{i});
            axis image;
            axis off;
        end
        pause;
    end

    hold off;

else strcmp(visType, 'groundTruth')
    conf_vec = C(classNum, :);
    
    conf_classes = find(conf_vec);

    conf_idx = intersect(find(gt_labels == classNum), find(ismember(pr_labels, conf_classes)));
    img_list = cell(sum(conf_vec), 1); % sum(conf_vec) should equal to length(conf_idx)
    img_list = DS.TI(conf_idx);

    numConfusion = length(conf_idx);
    numIters = ceil(numConfusion/(numRows*numCols));

    fig = figure;
    hold on;
    set(fig, 'Position', [0, 700, 1300, 1000], 'name', figTitle);

    for iter=1:numIters
        fprintf('iter %d/%d\n', iter, numIters);
        clf('reset');
        for i=(iter-1)*numRows*numCols+1:iter*numRows*numCols
            if i > length(conf_idx), break; end
            
            plotPos = mod(i, numRows*numCols);
            if plotPos == 0, plotPos = numRows*numCols; end

            subplot(numRows, numCols, plotPos);
            imagesc(img_list{i});
            axis image;
            axis off;
        end
        pause;
    end

    hold off;
end