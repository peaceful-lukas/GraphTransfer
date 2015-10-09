function sTriplets = local_sampleStructurePreservingTriplets(DS, W, U, param, trainTargetClasses, targetProtoIdx)

% (k, k', l), the indices regard the actual permutation in U

protoStartIdx = [0; cumsum(param.numPrototypes)];


num_sTriplets = size(param.sTriplets, 1);
sTriplets = [];
if num_sTriplets > 0
    sTriplets = param.sTriplets(randperm(num_sTriplets, min(param.s_batchSize, num_sTriplets)), :);
    target_idx = find(ismember(sTriplets(:, 1), targetProtoIdx));
    sTriplets = sTriplets(target_idx, :);

    loss = param.s_lm + diag( U(:, sTriplets(:, 1))' * (U(:, sTriplets(:, 3)) - U(:, sTriplets(:, 2))) );
    valids = find(loss > 0);
    sTriplets = sTriplets(valids, :);
end
