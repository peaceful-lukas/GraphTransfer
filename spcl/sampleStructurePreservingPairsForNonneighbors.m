function [sp_dist_pairs total_num_sp_dist_pairs] = sampleStructurePreservingPairsForNonneighbors(U, param)

sp_dist_pairs = [];

for classNum=1:param.numClasses
    A_c = param.nnGraphs{classNum};
    [k_vec, l_vec] = find(triu(~A_c - eye(size(A_c))));
    sp_dist_pairs = [sp_dist_pairs; k_vec, l_vec];
end

total_num_sp_dist_pairs = size(sp_dist_pairs, 1);

sp_dist_err_vec = param.sp_dist_bound - sum((U(:, sp_dist_pairs(:, 1)) - U(:, sp_dist_pairs(:, 2))).^2, 1);
viol_vec = find(sp_dist_err_vec > 0);

sp_dist_pairs = sp_dist_pairs(viol_vec, :);
