function [sTriplets total_num_sTriplets] = sampleStructurePreservingTriplets(U, param)
% (k, k', l)


sTriplets = [];
dist_margin_vec = [];

protoStartIdx = cumsum([0; param.numPrototypes]);

for classNum=1:param.numClasses
    U_c = U(:, protoStartIdx(classNum)+1:protoStartIdx(classNum+1));
    D_c = pdist(U_c');
    D_c = squareform(D_c);
    A_c = param.nnGraphs{classNum};
    D_nearest = D_c.*A_c;
    [D_nearest_max, D_nearest_max_idx] = max(D_nearest, [], 2);


    [k_vec, l_vec] = find(triu(~A_c - eye(param.numPrototypes(classNum))));
    k_prime_vec = D_nearest_max_idx(k_vec);
    dist_margin_vec_c = D_nearest_max(k_prime_vec);


    sTriplets = [sTriplets; k_vec, k_prime_vec, l_vec];
    dist_margin_vec = [dist_margin_vec; dist_margin_vec_c];
end

total_num_sTriplets = size(sTriplets, 1);

sErr_vec = 2 * diag(U(:, sTriplets(:, 1))'*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3))))    ...
            - diag(U(:, sTriplets(:, 2))'*U(:, sTriplets(:, 2)))                                ...
            + diag(U(:, sTriplets(:, 3))'*U(:, sTriplets(:, 3)));

viol_vec = find(sErr_vec > 0);
sTriplets = sTriplets(viol_vec, :);










% sTriplets = [];
% proto_offset = [0; cumsum(param.numPrototypes)];

% for classNum=1:param.numClasses
%     A_c = param.nnGraphs{classNum};
    
%     [k_vec, l_vec] = find(triu(~A_c - eye(size(A_c))));
%     [~, k_prime_vec] = find(triu(A_c));
    
%     k_vec = k_vec + proto_offset(classNum);
%     l_vec = l_vec + proto_offset(classNum);
%     k_prime_vec = k_prime_vec + proto_offset;

%     sTriplets = [sTriplets; k_vec, k_prime_vec, l_vec];
% end

% total_num_sTriplets = size(sTriplets, 1);

% s_err_vec = param. - sum((U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 2))).^2, 1);
% viol_vec = find(s_err_vec > 0);

% sTriplets = sTriplets(viol_vec, :);

