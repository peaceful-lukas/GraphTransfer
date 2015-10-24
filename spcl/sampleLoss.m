function loss = sampleLoss(DS, W, U, param)

X = DS.D;

cl_same_pairs = sampleClassificationPairsForSameClass(DS, W, U, param);
cl_diff_pairs = sampleClassificationPairsForDiffClass(DS, W, U, param);
[sp_near_pairs total_num_sp_near_pairs] = sampleStructurePreservingPairsForNeighbors(U, param);
[sp_dist_pairs total_num_sp_dist_pairs] = sampleStructurePreservingPairsForNonneighbors(U, param);

num_cl_same_pairs = size(cl_same_pairs, 1);
num_cl_diff_pairs = size(cl_diff_pairs, 1);
num_sp_near_pairs = size(sp_near_pairs, 1);
num_sp_dist_pairs = size(sp_dist_pairs, 1);

cl_same_err = 0;
num_cl_same_viol = 0;
if num_cl_same_pairs > 0
    cl_same_err_vec = sum((W*X(:, cl_same_pairs(:, 1)) - U(:, cl_same_pairs(:, 2))).^2, 1) - param.cl_same_bound;
    viol_vec = find(cl_same_err_vec > 0);
    num_cl_same_viol = length(viol_vec);
    cl_same_err = sum(cl_same_err_vec(viol_vec))/param.cl_same_batchSize;
end

cl_diff_err = 0;
num_cl_diff_viol = 0;
if num_cl_diff_pairs > 0
    cl_diff_err_vec = param.cl_diff_bound - sum((W*X(:, cl_diff_pairs(:, 1)) - U(:, cl_diff_pairs(:, 2))).^2, 1);
    viol_vec = find(cl_diff_err_vec > 0);
    num_cl_diff_viol = length(viol_vec);
    cl_diff_err = sum(cl_diff_err_vec(viol_vec))/param.cl_diff_batchSize;
end

sp_near_err = 0;
num_sp_near_viol = 0;
if num_sp_near_pairs > 0
    sp_near_err_vec = sum((U(:, sp_near_pairs(:, 1)) - U(:, sp_near_pairs(:, 2))).^2, 1) - param.sp_near_bound;
    viol_vec = find(sp_near_err_vec > 0);
    num_sp_near_viol = length(viol_vec);
    sp_near_err = sum(sp_near_err_vec(viol_vec))/total_num_sp_near_pairs;
end

sp_dist_err = 0;
num_sp_dist_viol = 0;
if num_sp_dist_pairs > 0
    sp_dist_err_vec = param.sp_dist_bound - sum((U(:, sp_dist_pairs(:, 1)) - U(:, sp_dist_pairs(:, 2))).^2, 1);
    viol_vec = find(sp_dist_err_vec > 0);
    num_sp_dist_viol = length(viol_vec);
    sp_dist_err = sum(sp_dist_err_vec(viol_vec))/total_num_sp_dist_pairs;
end


loss = param.bal_cl*(cl_same_err + cl_diff_err) + param.bal_sp*(sp_near_err + sp_dist_err) + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
% fprintf('cl: %d / %d | sp: %d / %d | W: %d | U: %f ', num_cl_same_viol, num_cl_diff_viol, num_sp_near_viol, num_sp_dist_viol, norm(W, 'fro')/size(W, 2), norm(U, 'fro')/size(U, 2));
fprintf('clsV: %d / cldV: %d / spnV: %d / spdV: %d / clsE: %f / cldE: %f / spnE: %f / spdE: %f / normW: %f / normU: %f / ', num_cl_same_viol, num_cl_diff_viol, num_sp_near_viol, num_sp_dist_viol, cl_same_err, cl_diff_err, sp_near_err, sp_dist_err, norm(W, 'fro')/size(W, 2), norm(U, 'fro')/size(U, 2));
