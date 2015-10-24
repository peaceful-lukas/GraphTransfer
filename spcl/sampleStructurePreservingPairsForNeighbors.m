function [sp_near_pairs total_num_sp_near_pairs] = sampleStructurePreservingPairsForNeighbors(U, param)

sp_near_pairs = [];
proto_offset = [0; cumsum(param.numPrototypes)];

for classNum=1:param.numClasses
    A_c = param.nnGraphs{classNum};

    [k_vec, k_prime_vec] = find(triu(A_c));
    k_vec = k_vec + proto_offset(classNum);
    k_prime_vec = k_prime_vec + proto_offset(classNum);

    sp_near_pairs = [sp_near_pairs; k_vec, k_prime_vec];
end

total_num_sp_near_pairs = size(sp_near_pairs, 1);

sp_near_err_vec = sum((U(:, sp_near_pairs(:, 1)) - U(:, sp_near_pairs(:, 2))).^2, 1) - param.sp_near_bound;
viol_vec = find(sp_near_err_vec > 0);

sp_near_pairs = sp_near_pairs(viol_vec, :);
