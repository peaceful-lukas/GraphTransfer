function W = learnW(DS, W, U, param)

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterW
    cTriplets = sampleClassificationTriplets(DS, W, U, param);
    pPairs = samplePullingPairs(DS, W, U, param);
    dW = computeGradient(DS, W, U, cTriplets, pPairs, param);
    W = update(W, dW, param);

    if ~mod(n, dispCycle)
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
function dW = computeGradient(DS, W, U, cTriplets, pPairs, param)

X = DS.D;
num_cTriplets = size(cTriplets, 1);
num_pPairs = size(pPairs, 1);

c_dW = zeros(size(W));
if num_cTriplets > 0
    c_dW = (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))) * X(:, cTriplets(:, 1))';
    c_dW = c_dW/param.c_batchSize;
    c_dW = c_dW/norm(c_dW, 'fro');
end

p_dW = zeros(size(W));
if num_pPairs > 0
    p_dW = W*X(:, pPairs(:, 1))*X(:, pPairs(:, 1))' - U(:, pPairs(:, 2))*X(:, pPairs(:, 1))';
    p_dW = 2*p_dW/param.p_batchSize;
    p_dW = p_dW/norm(p_dW, 'fro');
end

bal_c = param.bal_c/(param.bal_c + param.bal_p);
bal_p = param.bal_p/(param.bal_c + param.bal_p);

dW = bal_c*c_dW + bal_p*p_dW + param.lambda_W*W;

