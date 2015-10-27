function bargraphTransferResult(DS, W, U_after, param_after, U_before, param_before)

clsnames = stringifyClasses(param_before.dataset);

acc_list = [];
for cls = 1:param_after.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W, U_before, param_before);
    new_acc = getNewAccuracy(cls, DS, W, U_after, param_after);
    acc_list = [acc_list; orig_acc new_acc];
end

drawOverlappedBargraph(acc_list, param_after);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function drawOverlappedBargraph(acc_list, param)

figure;
hold on;

min_acc_list = min(acc_list, [], 2);
sub_acc_list = acc_list-repmat(min_acc_list, 1, 2);
acc_list = [min_acc_list sub_acc_list];

bar_handle = bar(acc_list, 'stacked');
bar_handle(1).FaceColor = [0.8 0.8 0.8]; % gray
bar_handle(2).FaceColor = 'r'; % before
bar_handle(3).FaceColor = 'y'; % after

legend('overlapped', 'before', 'after');


axis([0 param.numClasses+1 0 1]);
category_labels = stringifyClasses(param.dataset);
xticklabel_rotate([1:param.numClasses], 45, category_labels);

hold off;
drawnow;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function orig_acc = getOriginalAccuracy(cls, DS, W0, U0, param0)

% %%%%%% Disp Accuracy
cumNumProto = cumsum(param0.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W0'*U0, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param0.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
orig_acc = numel(find(classified == cls))/numel(find(DS.TL == cls));
% fprintf('ORIGINAL accuracy for class %d ----> %.4f\n', cls, orig_acc);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function new_acc = getNewAccuracy(cls, DS, W_new, U_new, param_new)

cumNumProto = cumsum(param_new.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W_new'*U_new, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param_new.numClasses
    t = find(classified_raw <= cumNumProto(c));
    classified(t) = c;
    classified_raw(t) = Inf;
end
new_acc = numel(find(classified == cls))/numel(find(DS.TL == cls));
% fprintf('TRANSFER accuracy for class %d ----> %.4f\n', cls, new_acc);

