function W = learnW(DS, W, U, param)

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterW
    mTriplets = sampleMembershipTriplets(DS, W, U, param);
    pPairs = samplePullingPairs(DS, W, U, param);
    dW = computeGradient(DS, W, U, mTriplets, pPairs, param);
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
function dW = computeGradient(DS, W, U, mTriplets, pPairs, param)

X = DS.D;
num_mTriplets = size(mTriplets, 1);
num_pPairs = size(pPairs, 1);

m_dW = zeros(size(W));
if num_mTriplets > 0
    m_dW = (U(:, mTriplets(:, 3)) - U(:, mTriplets(:, 2))) * X(:, mTriplets(:, 1))';
    m_dW = m_dW/param.m_batchSize;
end

p_dW = zeros(size(W));
if num_pPairs > 0
    p_dW = U(:, pPairs(:, 2))*X(:, pPairs(:, 1))';
    p_dW = p_dW/param.p_batchSize;
end

bal_m = param.bal_m/(param.bal_m + param.bal_p);
bal_p = param.bal_p/(param.bal_m + param.bal_p);

dW = bal_m*m_dW + bal_p*p_dW + param.lambda_W*W;

