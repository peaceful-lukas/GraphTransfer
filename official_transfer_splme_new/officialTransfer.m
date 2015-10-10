function [U_test, param_test, inferred_idx] = officialTransfer(U_train, U_test, trClass, teClass, scale_alpha, param_train, param_test)

% Graph Matching
[matched_pairs, unmatched, U_trClass, U_teClass] = graphMatching(U_train, U_test, param_train, param_test, trClass, teClass);

% Transfer

if numel(unmatched) > 0
    fprintf('\n\nTransfer begins!!\n\n');
    newPrototypes = computeNewPrototypes(U_train, U_test, param_train, param_test, matched_pairs, unmatched, U_trClass, U_teClass, trClass, teClass, scale_alpha);
    [U_test, param_test, inferred_idx] = updatePrototypes(U_test, param_train, param_test, newPrototypes, trClass, teClass, matched_pairs, unmatched);
else
    fprintf('\n\nNo transfer...\n\n');
end






function [matched_pairs, unmatched, U_trClass, U_teClass] = graphMatching(U_train, U_test, param_train, param_test, trClass, teClass)


numTrainProto = param_train.numPrototypes(trClass);
trainProtoStartIdx = [0; cumsum(param_train.numPrototypes)];
U_trClass = U_train(:, trainProtoStartIdx(trClass)+1:trainProtoStartIdx(trClass+1));

numTestProto = param_test.numPrototypes(teClass);
testProtoStartIdx = [0; cumsum(param_test.numPrototypes)];
U_teClass = U_test(:, testProtoStartIdx(teClass)+1:testProtoStartIdx(teClass+1));

simMatrix = U_trClass'*U_teClass;
sim_scores = sort(simMatrix(:), 'descend');

%%%%%% Graph Matching - RRWM
param_gm.maxIterGM = 10;
param_gm.match_thrsh = sim_scores(min(numTrainProto, numTestProto));
param_gm.match_sim_thrsh = sim_scores(max(numTrainProto, numTestProto));
param_gm.knn1 = 3;
param_gm.knn2 = 4;
param_gm.voting_alpha = 10;


[X_sol, cand_matches, score_GM] = progGM(U_trClass, U_teClass, param_gm);
matched_pairs = cand_matches(find(X_sol), :);

unmatched = 1:numTrainProto;
unmatched(matched_pairs(:, 1)) = [];


function newPrototypes = computeNewPrototypes(U_train, U_test, param_train, param_test, matched_pairs, unmatched, U_trClass, U_teClass, trClass, teClass, scale_alpha)

numMatched = size(matched_pairs, 1);
newPrototypes = [];

for um_idx=1:length(unmatched)
    fromProto = unmatched(um_idx);
    toProto = zeros(param_test.lowDim, 1);
    
    for n=1:numMatched
        toProto = toProto + U_teClass(:, matched_pairs(n, 2)) - scale_alpha*(U_trClass(:, matched_pairs(n, 1)) - U_trClass(:, fromProto));
    end
    toProto = toProto/numMatched;
    
    newPrototypes = [newPrototypes toProto];
end


function [U_test, param_test, inferred_idx] = updatePrototypes(U_test, param_train, param_test, newPrototypes, trClass, teClass, matched, unmatched)
% 1. add transferred prototypes into U
% 2. update the number of prototypes of the class
% 3. update the knn-graph of the class

testProtoStartIdx = [0; cumsum(param_test.numPrototypes)];
U_test = [U_test(:, 1:testProtoStartIdx(teClass+1)) newPrototypes U_test(:, testProtoStartIdx(teClass+1)+1:end)];

numNewPrototypes = size(newPrototypes, 2);
inferred_idx = 1:numNewPrototypes;
inferred_idx = inferred_idx + testProtoStartIdx(teClass+1);

param_test.numPrototypes(teClass) = param_test.numPrototypes(teClass) + length(inferred_idx);



A_train = param_train.knnGraphs{trClass};
A_test = param_test.knnGraphs{teClass};

A_test_new = zeros(size(A_test, 1) + numNewPrototypes);
A_test_new(1:size(A_test, 1), 1:size(A_test, 1)) = A_test;

n = size(A_test, 1);
for i=1:size(matched, 1)
    for j=1:length(unmatched)
        if A_train(matched(i, 1), unmatched(j)) == 1
            A_test_new(matched(i, 2), n+j) = 1;
        end
    end
end

A_test_new(n+1:end, n+1:end) = A_train(unmatched, unmatched);

param_test.knnGraphs{teClass} = A_test_new;







