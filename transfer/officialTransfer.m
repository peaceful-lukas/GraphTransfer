function [U_test numPrototypes_test] = officialTransfer(DS, W, U, numPrototypes, prototypes, trClass, teClass, scale_alpha)

graphMatching(U, numPrototypes, prototypes, trClass, teClass);

%%%%%%%% Transfer Prototypes
new_numPrototypes = param_new.numPrototypes;

unmatched = 1:param0.numPrototypes(c1);
unmatched(matched_pairs(:, 1)) = [];

if numel(unmatched) > 0
    fprintf('\n\nTransfer begins!!\n\n');
    transferred_prototypes = [];
    for um_idx=1:length(unmatched)
        target = unmatched(um_idx);
        transferred = zeros(param0.lowDim, 1);
        for n=1:numMatched
            transferred = transferred + U_c2(:, matched_pairs(n, 2)) - scale_alpha * (U_c1(:, matched_pairs(n, 1)) - U_c1(:, target));
        end
        transferred = transferred/numMatched;
        transferred_prototypes = [transferred_prototypes transferred];
    end

    [U_new param_new inferred_idx] = updatePrototypes(U_new, transferred_prototypes, c1, c2, matched_pairs, unmatched, param_new);


    dispAccuracies(DS, W_new, U_new, W0, U0, param_new, param0);
    trainTargetClasses = getClassesToBeLocallyTrained(DS, W_new, U_new, W0, U0, param_new, param0);

    % fprintf('Transfer Result : \n');
    % fprintf('BEFORE TRANSFER >>\n');
    % [~, accuracy] = dispAccuracy(param0.method, DS, W0, U0, param0);
    % fprintf('AFTER TRANSFER >>\n');
    % [~, accuracy] = dispAccuracy(param0.method, DS, W_new, U_new, param_new);
    % fprintf('\n\n');
    % bargraphTransferResult(DS, W_new, U_new, param_new, W0, U0, param0);

else
    fprintf('\n\nNo transfer...\n\n');
    inferred_idx = [];
    trainTargetClasses = [];
end


function graphMatching(U, numPrototypes, prototypes, trClass, teClass)



trNumProto = numPrototypes.train(trClass);
teNumProto = numPrototypes.test(teClass);
U_tr = U_train(:, );
U_te = U_test(:, protoStartIdx(c2)+1:protoStartIdx(c2+1));

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




