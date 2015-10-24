function W = learnW(DS, W, U, param)

dispCycle = 100;
n = 1;

X = DS.D;

tic;
while n <= param.maxIterW
    dW = computeGradient(DS, W, U, param);
    W = update(W, dW, param);

    if ~mod(n, dispCycle) || n == 1
        timeElapsed = toc;
        fprintf('W%d) ', n);
        loss = sampleLoss(DS, W, U, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end



% update
function W = update(W, dW, param)

W = W - param.lr_W * dW;



% gradient computation
function dW = computeGradient(DS, W, U, param)

X = DS.D;

cl_same_pairs = sampleClassificationPairsForSameClass(DS, W, U, param);
cl_diff_pairs = sampleClassificationPairsForDiffClass(DS, W, U, param);

num_cl_same_pairs = size(cl_same_pairs, 1);
num_cl_diff_pairs = size(cl_diff_pairs, 1);

cl_same_dW = zeros(size(W));
if num_cl_same_pairs > 0
    cl_same_dW = 2*(W*X(:, cl_same_pairs(:, 1)) - U(:, cl_same_pairs(:, 2)))*X(:, cl_same_pairs(:, 1))';
    cl_same_dW = cl_same_dW/param.cl_same_batchSize;
end

cl_diff_dW = zeros(size(W));
if num_cl_diff_pairs > 0
    cl_diff_dW = -2*(W*X(:, cl_diff_pairs(:, 1)) - U(:, cl_diff_pairs(:, 2)))*X(:, cl_diff_pairs(:, 1))';
    cl_diff_dW = cl_diff_dW/param.cl_diff_batchSize;
end

dW = cl_same_dW + cl_diff_dW + param.lambda_W*W;
dW = dW/size(W, 2);
