function [U_new param_new matched_pairs inferred_idx trainTargetClasses, score_GM] = transfer(DS, W_new, U_new, W0, U_prev, c1, c2, scale_alpha, param_new, param_prev)
% TRANSFER
%    transfer class prototypes ( c1 ---> c2 )
%    All the unmatched prototypes are transferred to the class c2



numProto_c1 = param_prev.numPrototypes(c1);
numProto_c2 = param_new.numPrototypes(c2);
protoStartIdx = [0; cumsum(param_prev.numPrototypes)];
U_c1 = U_prev(:, protoStartIdx(c1)+1:protoStartIdx(c1+1));
U_c2 = U_new(:, protoStartIdx(c2)+1:protoStartIdx(c2+1));

simMatrix = U_c1'*U_c2;
sim_scores = sort(simMatrix(:), 'descend');

%%%%%% Graph Matching - RRWM
param_gm.maxIterGM = 10;
param_gm.match_thrsh = sim_scores(min(numProto_c1, numProto_c2));
param_gm.match_sim_thrsh = sim_scores(max(numProto_c1, numProto_c2));
param_gm.knn1 = 3;
param_gm.knn2 = 4;
param_gm.voting_alpha = 10;


[X_sol, cand_matches, score_GM] = progGM(U_c1, U_c2, param_gm);
matched_pairs = cand_matches(find(X_sol), :);
numMatched = size(matched_pairs, 1);


% TRANSFER DIRECTION :   c1 -----------> c2 

new_numPrototypes = param_new.numPrototypes;

unmatched = 1:param_prev.numPrototypes(c1);
unmatched(matched_pairs(:, 1)) = [];

if numel(unmatched) > 0
    fprintf('\n\nTransfer begins!!\n\n');
    transferred_prototypes = [];
    for um_idx=1:length(unmatched)
        target = unmatched(um_idx);
        transferred = zeros(param_prev.lowDim, 1);
        for n=1:numMatched
            transferred = transferred + U_c2(:, matched_pairs(n, 2)) - scale_alpha * (U_c1(:, matched_pairs(n, 1)) - U_c1(:, target));
        end
        transferred = transferred/numMatched;
        transferred_prototypes = [transferred_prototypes transferred];
    end

    [U_new param_new inferred_idx] = updatePrototypes(U_new, transferred_prototypes, c1, c2, matched_pairs, unmatched, param_new);

    official_dispAccuracy(DS, W0, U_new, param_new, U_prev, param_prev);
    % dispAccuracies(DS, W_new, U_new, W0, U_prev, param_new, param_prev);
    bargraphTransferResult(DS, W0, U_new, param_new, U_prev, param_prev);
    trainTargetClasses = getClassesToBeLocallyTrained(DS, W_new, U_new, W0, U_prev, param_new, param_prev);

    fprintf('Transfer Result : \n');
    fprintf('BEFORE TRANSFER >>\n');
    [~, accuracy] = dispAccuracy(param_prev.method, DS, W0, U_prev, param_prev);
    fprintf('AFTER TRANSFER >>\n');
    [~, accuracy] = dispAccuracy(param_prev.method, DS, W_new, U_new, param_new);
    fprintf('\n\n');

else
    fprintf('\n\nNo transfer...\n\n');
    inferred_idx = [];
    trainTargetClasses = [];
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U_new param_new inferred_idx] = updatePrototypes(U_new, transferred_prototypes, c1, c2, matched, unmatched, param_new)
% 1. add transferred prototypes into U
% 2. update the number of prototypes of the class
% 3. update the knn-graph of the class

U_new = [U_new(:, 1:sum(param_new.numPrototypes(1:c2))) transferred_prototypes U_new(:, sum(param_new.numPrototypes(1:c2))+1:end)];

cumSumProto = cumsum(param_new.numPrototypes);
inferred_idx = 1:length(unmatched);
inferred_idx = inferred_idx + cumSumProto(c2);

param_new.numPrototypes(c2) = param_new.numPrototypes(c2) + length(unmatched);



A1 = param_new.knnGraphs{c1};
A2 = param_new.knnGraphs{c2};
A2_new = zeros(size(A2, 1)+length(unmatched));
A2_new(1:size(A2, 1), 1:size(A2, 1)) = A2;

n = size(A2, 1);
for i=1:size(matched, 1)
    for j=1:length(unmatched)
        if A1(matched(i, 1), unmatched(j)) == 1
            A2_new(matched(i, 2), n+j) = 1;
        end
    end
end

A2_new(n+1:end, n+1:end) = A1(unmatched, unmatched);

param_new.knnGraphs{c2} = A2_new;







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function trainTargetClasses = getClassesToBeLocallyTrained(DS, W_new, U_new, W0, U_prev, param_new, param_prev)
trainTargetClasses = [];
for cls = 1:param_new.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W0, U_prev, param_prev);
    new_acc = getNewAccuracy(cls, DS, W_new, U_new, param_new);
    if new_acc == orig_acc
        continue;
    else
        trainTargetClasses = [trainTargetClasses; cls];
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function acc_list = dispAccuracies(DS, W_new, U_new, W0, U_prev, param_new, param_prev)
clsnames = stringifyClasses(param_prev.dataset);

acc_list = [];
for cls = 1:param_new.numClasses
    orig_acc = getOriginalAccuracy(cls, DS, W0, U_prev, param_prev);
    new_acc = getNewAccuracy(cls, DS, W_new, U_new, param_new);
    acc_list = [acc_list; orig_acc new_acc];
    fprintf('Accuracy (%s) : %.4f ----> %.4f ', clsnames{cls}, orig_acc, new_acc);
    if orig_acc > new_acc, fprintf('(down)\n');
    elseif orig_acc < new_acc, fprintf('(UP)\n');
    else fprintf('\n');
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function orig_acc = getOriginalAccuracy(cls, DS, W0, U_prev, param_prev)

% %%%%%% Disp Accuracy
cumNumProto = cumsum(param_prev.numPrototypes);
classIdx = find(DS.TL == cls);
class_feat = DS.T(:, classIdx);
[~, classified_raw]= max(class_feat'*W0'*U_prev, [], 2);
classified = zeros(numel(classified_raw), 1);
for c = 1:param_prev.numClasses
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

