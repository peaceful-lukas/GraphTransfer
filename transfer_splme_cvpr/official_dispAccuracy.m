function official_dispAccuracy(DS, W, U_new, param_new, U_prev, param_prev)


[U_prev_test, param_prev_test] = getOfficialTestset(U_prev, param_prev);
[U_new_test, param_new_test] = getOfficialTestset(U_new, param_new);

test_classes = [6, 14, 15, 18, 24, 25, 34, 39, 42, 48];
clsnames = stringifyClasses(param_prev.dataset);
clsnames = clsnames(test_classes);

acc_list = [];
for cls = 1:param_prev_test.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W, U_prev_test, param_prev_test);
    new_acc = getNewAccuracy(cls, DS, W, U_new_test, param_new_test);
    
    acc_list = [acc_list; orig_acc new_acc];
    fprintf('Accuracy (%s) : %.4f ----> %.4f ', clsnames{cls}, orig_acc, new_acc);
    if orig_acc > new_acc, fprintf('(down)\n');
    elseif orig_acc < new_acc, fprintf('(UP)\n');
    else fprintf('\n');
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U_test param_test] = getOfficialTestset(U_all, param_all)

train_classes = 1:50;
test_classes = [6, 14, 15, 18, 24, 25, 34, 39, 42, 48];
train_classes(find(ismember(train_classes, test_classes))) = [];

protoStartIdx = [0; cumsum(param_all.numPrototypes)];

U_test = [];
for cls=test_classes
    testProtoIdx = protoStartIdx(cls)+1:protoStartIdx(cls+1);
    U_test = [U_test U_all(:, testProtoIdx)];
end

param_test = param_all;
param_test.numClasses = length(test_classes);
param_test.numPrototypes = param_all.numPrototypes(test_classes);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function orig_acc = getOriginalAccuracy(cls, DS, W, U_prev_test, param_prev_test)

% %%%%%% Disp Accuracy
cumNumProto = cumsum(param_prev_test.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);

[~, classified_raw]= max(class_feat'*W'*U_prev_test, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param_prev_test.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
orig_acc = numel(find(classified == cls))/numel(find(DS.TL == cls));
% fprintf('ORIGINAL accuracy for class %d ----> %.4f\n', cls, orig_acc);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_acc = getNewAccuracy(cls, DS, W, U_new_test, param_new_test)

cumNumProto = cumsum(param_new_test.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W'*U_new_test, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param_new_test.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
new_acc = numel(find(classified == cls))/numel(find(DS.TL == cls));
% fprintf('TRANSFER accuracy for class %d ----> %.4f\n', cls, new_acc);

