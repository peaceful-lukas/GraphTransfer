function [tPairs str_tPairs S] = officialTransferPairs(U, param)

protoStartIdx = [0; cumsum(param.numPrototypes)];

S = zeros(10, 40); % Similarity between class prototype distributions
train_classes = 1:50;
test_classes = [6, 14, 15, 18, 24, 25, 34, 39, 42, 48];
train_classes(find(ismember(train_classes, test_classes))) = [];

U_norm = normc(U);
for i=1:length(test_classes)
    for j=1:length(train_classes)
        % maximum score
        teClass = test_classes(i);
        trClass = train_classes(j);
        S(i, j) = max(max(U_norm(:, protoStartIdx(trClass)+1:protoStartIdx(trClass+1))'*U_norm(:, protoStartIdx(teClass)+1:protoStartIdx(teClass+1))));
    end
end


[S_sorted, sorted_idx] = sort(S, 2, 'descend');

transfer_classes = find(S_sorted(:, 1) > 0.7); % classes to be transferred
tPairs = zeros(length(transfer_classes), 2); % 10 = # of test classes
tPairs(:, 1) = sorted_idx(transfer_classes, 1);
tPairs(:, 2) = test_classes(transfer_classes);

% stringify class names
clsnames = stringifyClasses(param.dataset);
str_tPairs = cell(size(tPairs));
for i=1:size(tPairs, 1)
    str_tPairs{i, 1} = clsnames{tPairs(i, 1)};
    str_tPairs{i, 2} = clsnames{tPairs(i, 2)};
end