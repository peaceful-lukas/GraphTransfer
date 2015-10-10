function officialDispAccuracy(DS_official, W, U_test_new, param_test_new, U_test_prev, param_test_prev, clsnames)

for i= 1:param_test_prev.numClasses
    classNum = param_test_prev.test_classes(i);
    orig_acc = getClassAccuracy(DS_official, W, U_test_prev, param_test_prev, classNum);
    new_acc = getClassAccuracy(DS_official, W, U_test_new, param_test_new, classNum);
    
    fprintf('Accuracy (%-20s) : %.4f ----> %.4f ', clsnames{i}, orig_acc, new_acc);
    if orig_acc > new_acc, fprintf('(down)\n');
    elseif orig_acc < new_acc, fprintf('(UP)\n');
    else fprintf('\n');
    end
end

fprintf('Before accuracy : %.4f\n', getOverallAccuracy(DS_official, W, U_test_prev, param_test_prev));
fprintf('After  accuracy : %.4f\n', getOverallAccuracy(DS_official, W, U_test_new, param_test_new));


function orig_acc = getClassAccuracy(DS, W, U, param, classNum)

cumNumProto = cumsum(param.numPrototypes);
classIdx = find(DS.TL == classNum);
[~, classified_raw]= max(DS.T(:, classIdx)'*W'*U, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = param.test_classes(c);
    classified_raw(t) = Inf;
end
orig_acc = numel(find(classified == classNum))/numel(find(DS.TL == classNum));



function overall_acc = getOverallAccuracy(DS, W, U, param)


cumNumProto = cumsum(param.numPrototypes);
for classNum=1:param.numClasses
    [~, classified_raw]= max(DS.T'*W'*U, [], 2);
    classified = zeros(numel(classified_raw), 1);
    for c = 1:param.numClasses
        t = find(classified_raw <= cumNumProto(c));
        classified(t) = param.test_classes(c);
        classified_raw(t) = Inf;
    end
end

overall_acc = numel(find(DS.TL == classified))/numel(DS.TL);