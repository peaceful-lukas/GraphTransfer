function [tPairs S] = officialTransferPairs(U_train, U_test, param_train, param_test)
% function [tPairs str_tPairs S] = officialTransferPairs(U, param)

trainProtoStartIdx = [0; cumsum(param_train.numPrototypes)];
testProtoStartIdx = [0; cumsum(param_test.numPrototypes)];

train_classes = param_train.train_classes;
test_classes = param_test.test_classes;

U_train_norm = normc(U_train);
U_test_norm = normc(U_test);

S = zeros(10, 40); % Similarity between class prototype distributions
for teClass=1:length(test_classes)
    for trClass=1:length(train_classes)
        % maximum score
        S(teClass, trClass) = max(max(U_train_norm(:, trainProtoStartIdx(trClass)+1:trainProtoStartIdx(trClass+1))'*U_test_norm(:, testProtoStartIdx(teClass)+1:testProtoStartIdx(teClass+1))));
    end
end

[S_sorted, sorted_idx] = sort(S, 2, 'descend');
transfer_classes = find(S_sorted(:, 1) > 0.5);
tPairs = zeros(length(transfer_classes), 2);
tPairs(:, 1) = sorted_idx(transfer_classes, 1); % train classes (from)
tPairs(:, 2) = transfer_classes; % test classes (to)