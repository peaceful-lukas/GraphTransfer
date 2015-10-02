function result = findExamples(exampleType, classNum, DS, W, U, param, W_new, U_new, param_new, visualize)

result = [];

switch(exampleType)
case 'gotTrueAfterTransfer'
    result = gotTrueAfterTransfer(DS, classNum, W, U, param, W_new, U_new, param_new, visualize);

case 'gotFalseAfterTransfer'
    
end




function gotTrue = gotTrueAfterTransfer(DS, classNum, W, U, param, W_new, U_new, param_new, visualize)

classIdx = find(DS.DL == classNum);

% before transfer
cumNumProto = cumsum(param.numPrototypes);
[~, predicted_raw] = max(DS.D'*W'*U, [], 2);
predicted = zeros(numel(predicted_raw), 1);
for c = 1:param.numClasses
    t = find(predicted_raw <= cumNumProto(c));
    predicted(t) = c;
    predicted_raw(t) = Inf;
end
wrong_predicts = find(DS.DL(classIdx) ~= predicted(classIdx));


% after transfer
cumNumProto_new = cumsum(param_new.numPrototypes);
[~, predicted_raw] = max(DS.D'*W_new'*U_new, [], 2);
predicted_new = zeros(numel(predicted_raw), 1);
for c = 1:param_new.numClasses
    t = find(predicted_raw <= cumNumProto_new(c));
    predicted_new(t) = c;
    predicted_raw(t) = Inf;
end
true_predicts = find(DS.DL(classIdx) == predicted_new(classIdx));

gotTrue = intersect(true_predicts, wrong_predicts);
gotTrue = classIdx(gotTrue);

if visualize
    numExamples = length(gotTrue);
    
    if numExamples > 0
        numRows = 5;
        numCols = 5;
        numSampleExamples = min(numExamples, numRows*numCols);
        sample_idx = gotTrue(randperm(numExamples, numSampleExamples));
        
        fig = figure;
        set(fig, 'Position', [0, 700, 1300, 1000]);
        for i=1:numSampleExamples
            subplot(numRows, numCols, i);
            imagesc(DS.DI{sample_idx(i)});
            axis image;
            axis off;
        end
    else
        fprintf('No such examples..\n');
    end
end


function gotFalse = gotFalseAfterTransfer(DS, classNum, W, U, param, W_new, U_new, param_new)

gotFalse = [];




