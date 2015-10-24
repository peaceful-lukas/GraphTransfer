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

cTriplets = sampleClassificationTriplets(DS, W, U, param);
mTriplets = sampleMembershipTriplets(DS, W, U, param);

num_cTriplets = size(cTriplets, 1);
num_mTriplets = size(mTriplets, 1);

c_dW = zeros(size(W));
if num_cTriplets > 0
    c_dW = -2*(U(:, cTriplets(:, 2)) - U(:, cTriplets(:, 3)))*bsxfun(@times, X(:, cTriplets(:, 1))', 1./param.numInstancesPerClass(DS.DL(cTriplets(:, 1))));
    c_dW = c_dW/param.c_batchSize;
end

m_dW = zeros(size(W));
if num_mTriplets > 0
    m_dW = -2*(U(:, mTriplets(:, 2)) - U(:, mTriplets(:, 3)))*bsxfun(@times, X(:, mTriplets(:, 1))', 1./param.numInstancesPerClass(DS.DL(mTriplets(:, 1))));
    m_dW = m_dW/param.m_batchSize;
end

dW = param.bal_c*c_dW + param.bal_m*m_dW + param.lambda_W*W;
dW = dW/size(W, 2);
