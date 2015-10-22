function sTriplets = sampleStructurePreservingTriplets(U, param)
% (k, k', l)


sTriplets = [];
dist_margin_vec = [];

protoStartIdx = cumsum([0; param.numPrototypes]);

for classNum=1:param.numClasses
    U_c = U(:, protoStartIdx(classNum)+1:protoStartIdx(classNum+1));
    D_c = pdist(U_c');
    D_c = squareform(D_c);
    A_c = param.knnGraphs{classNum};
    D_nearest = D_c.*A_c;
    [D_nearest_max, D_nearest_max_idx] = max(D_nearest, [], 2);

    [proto_idx, target_idx] = find(triu(~A_c - eye(param.numPrototypes(classNum))));
    max_nearest_idx = D_nearest_max_idx(proto_idx);
    dist_margin_vec_c = D_nearest_max(max_nearest_idx);

    sTriplets = [sTriplets; proto_idx, max_nearest_idx, target_vec];
    dist_margin_vec = [dist_margin_vec; dist_margin_vec_c];
end

loss_vec = 2 * diag(U(:, sTriplets(:, 1))'*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3))))    ...
            - diag(U(:, sTriplets(:, 2))'*U(:, sTriplets(:, 2)))                               ...
            + diag(U(:, sTriplets(:, 3))'*U(:, sTriplets(:, 3)));

valids = find(loss_vec > 0);
sTriplets = sTriplets(valids, :);


