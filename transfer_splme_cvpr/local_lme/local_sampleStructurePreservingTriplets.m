function [sTriplets total_num_sTriplets] = sampleStructurePreservingTriplets(U, param)
% (k, k', l)



sTriplets = [];
dist_margin_vec = [];

proto_offset = cumsum([0; param.numPrototypes]);

for classNum=1:param.numClasses
    U_c = U(:, proto_offset(classNum)+1:proto_offset(classNum+1));
    D_c = squareform(pdist(U_c'));
    A_c = param.nnGraphs{classNum};
    D_nearest = D_c.*A_c;
    [D_nearest_max, D_nearest_max_idx] = max(D_nearest, [], 2);


    [k_vec, l_vec] = find(~A_c - eye(param.numPrototypes(classNum)));
    k_prime_vec = D_nearest_max_idx(k_vec);
    dist_margin_vec_c = D_nearest_max(k_vec);

    k_vec = k_vec + proto_offset(classNum);
    l_vec = l_vec + proto_offset(classNum);
    k_prime_vec = k_prime_vec + proto_offset(classNum);

    sTriplets = [sTriplets; k_vec, k_prime_vec, l_vec];
    dist_margin_vec = [dist_margin_vec; dist_margin_vec_c];
end

total_num_sTriplets = size(sTriplets, 1);

sErr_vec = param.s_lm - 2 * diag(U(:, sTriplets(:, 1))'*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3))))    ...
            + diag(U(:, sTriplets(:, 2))'*U(:, sTriplets(:, 2)))                                ...
            - diag(U(:, sTriplets(:, 3))'*U(:, sTriplets(:, 3)));

viol_vec = find(sErr_vec > 0);
sTriplets = sTriplets(viol_vec, :);













% sTriplets = [];
% dist_margin_vec = [];

% proto_offset = cumsum([0; param.numPrototypes]);

% for classNum=1:param.numClasses
%     U_c = U(:, proto_offset(classNum)+1:proto_offset(classNum+1));
%     D_c = squareform(pdist(U_c'));
%     A_c = param.nnGraphs{classNum};
%     D_nearest = D_c.*A_c;
%     [D_nearest_max, D_nearest_max_idx] = max(D_nearest, [], 2);


%     [k_vec, l_vec] = find(~A_c - eye(param.numPrototypes(classNum)));
%     k_prime_vec = D_nearest_max_idx(k_vec);
%     dist_margin_vec_c = D_nearest_max(k_vec);

%     k_vec = k_vec + proto_offset(classNum);
%     l_vec = l_vec + proto_offset(classNum);
%     k_prime_vec = k_prime_vec + proto_offset(classNum);

%     sTriplets = [sTriplets; k_vec, k_prime_vec, l_vec];
%     dist_margin_vec = [dist_margin_vec; dist_margin_vec_c];
% end

% total_num_sTriplets = size(sTriplets, 1);

% sErr_vec = param.s_lm - 2 * diag(U(:, sTriplets(:, 1))'*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3))))    ...
%             + diag(U(:, sTriplets(:, 2))'*U(:, sTriplets(:, 2)))                                ...
%             - diag(U(:, sTriplets(:, 3))'*U(:, sTriplets(:, 3)));

% viol_vec = find(sErr_vec > 0);
% sTriplets = sTriplets(viol_vec, :);





