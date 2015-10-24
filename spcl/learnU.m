function U = learnU(DS, W, U, param)

dispCycle = 100;
n = 1;

WX = W*DS.D;
aux = eye(sum(param.numPrototypes));

tic;
while n <= param.maxIterU
    dU = computeGradient(DS, W, U, WX, aux, param);
    U = update(U, dU, param);

    if ~mod(n, dispCycle) || n == 1
        timeElapsed = toc;
        fprintf('U%d) ', n);
        loss = sampleLoss(DS, W, U, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end

% update
function U = update(U, dU, param)

U = U - param.lr_U * dU;



% gradient computation
function dU = computeGradient(DS, W, U, WX, aux, param)

cl_same_pairs = sampleClassificationPairsForSameClass(DS, W, U, param);
cl_diff_pairs = sampleClassificationPairsForDiffClass(DS, W, U, param);
[sp_near_pairs total_num_sp_near_pairs] = sampleStructurePreservingPairsForNeighbors(U, param);
[sp_dist_pairs total_num_sp_dist_pairs] = sampleStructurePreservingPairsForNonneighbors(U, param);

num_cl_same_pairs = size(cl_same_pairs, 1);
num_cl_diff_pairs = size(cl_diff_pairs, 1);
num_sp_near_pairs = size(sp_near_pairs, 1);
num_sp_dist_pairs = size(sp_dist_pairs, 1);

cl_same_dU = zeros(size(U));
if num_cl_same_pairs > 0
    cl_same_dU = -2*(WX(:, cl_same_pairs(:, 1)) - U(:, cl_same_pairs(:, 2)))*aux(param.protoAssign(cl_same_pairs(:, 1)), :);
    cl_same_dU = cl_same_dU/param.cl_same_batchSize;
end

cl_diff_dU = zeros(size(U));
if num_cl_diff_pairs > 0
    cl_diff_dU = 2*(WX(:, cl_diff_pairs(:, 1)) - U(:, cl_diff_pairs(:, 2)))*aux(param.protoAssign(cl_diff_pairs(:, 1)), :);
    cl_diff_dU = cl_diff_dU/param.cl_diff_batchSize;
end

sp_near_dU = zeros(size(U));
if num_sp_near_pairs > 0
    sp_near_dU = 2*(U(:, sp_near_pairs(:, 1)) - U(:, sp_near_pairs(:, 2)))*aux(sp_near_pairs(:, 1), :);
    sp_near_dU = sp_near_dU/total_num_sp_near_pairs;
end

sp_dist_dU = zeros(size(U));
if num_sp_dist_pairs > 0
    sp_dist_dU = -2*(U(:, sp_dist_pairs(:, 1)) - U(:, sp_dist_pairs(:, 2)))*aux(sp_dist_pairs(:, 1), :);
    sp_dist_dU = sp_dist_dU/total_num_sp_dist_pairs;
end

dU = param.bal_cl*(cl_same_dU + cl_diff_dU) + param.bal_sp*(sp_near_dU + sp_dist_dU) + param.lambda_U*U;
dU = dU/size(U, 2);

