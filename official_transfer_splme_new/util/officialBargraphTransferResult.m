function officialBargraphTransferResult(DS, W, U_test_new, param_test_new, U_test_prev, param_test_prev, clsnames)

acc_list = [];
for i=1:param_test_prev.numClasses
    classNum = param_test_prev.test_classes(i);
    orig_acc = getClassAccuracy(DS, W, U_test_prev, param_test_prev, classNum);
    new_acc = getClassAccuracy(DS, W, U_test_new, param_test_new, classNum);
    
    acc_list = [acc_list; orig_acc new_acc];
end

drawOverlappedBargraph(acc_list, param_test_prev, clsnames);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function drawOverlappedBargraph(acc_list, param, category_labels)

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


axis([0 11 0 1]);
xticklabel_rotate([1:10], 45, category_labels);

hold off;
drawnow;



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
