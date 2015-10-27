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

        
    end

    n = n + 1;
end



% update
function W = update(W, dW, param)

W = W - param.lr_W * dW;

if param.projected
    for i=1:size(W, 2)
        normWi = norm(W(:, i), 2);
        coef = abs(1 - param.lambda_W / normWi);
        W(:, i) = coef * W(:, i);
    end
end



% gradient computation
function dW = computeGradient(DS, W, U, param)

X = DS.D;

cTriplets = sampleClassificationTriplets(DS, W, U, param);
mTriplets = sampleMembershipTriplets(DS, W, U, param);
% mPairs = sampleMembershipUnaryPairs(DS, W, U, param);

num_cTriplets = size(cTriplets, 1);
num_mTriplets = size(mTriplets, 1);
% num_mPairs = size(mPairs, 1);


c_dW = zeros(size(W));
if num_cTriplets > 0
    % c_dW = -2*(U(:, cTriplets(:, 2)) - U(:, cTriplets(:, 3)))*bsxfun(@times, X(:, cTriplets(:, 1))', 1./param.numInstancesPerClass(DS.DL(cTriplets(:, 1))));
    c_dW = -2*(U(:, cTriplets(:, 2)) - U(:, cTriplets(:, 3)))*X(:, cTriplets(:, 1))';
    c_dW = c_dW/param.c_batchSize;
end

m_dW = zeros(size(W));
if num_mTriplets > 0
    m_dW = -2*(U(:, mTriplets(:, 2)) - U(:, mTriplets(:, 3)))*bsxfun(@times, X(:, mTriplets(:, 1))', 1./param.numInstancesPerClass(DS.DL(mTriplets(:, 1))));
    m_dW = m_dW/param.m_batchSize;
end

% mp_dW = zeros(size(W));
% if num_mPairs > 0
%     mp_dW = 2*(W*X(:, mPairs(:, 1)) - U(:, mPairs(:, 2)))*bsxfun(@times, X(:, mPairs(:, 1))', 1./param.numInstancesPerClass(DS.DL(mPairs(:, 1))));
%     mp_dW = mp_dW/param.m_batchSize;
% end

% dW = param.bal_c*c_dW + param.bal_m*(m_dW + mp_dW) + param.lambda_W*W;

if param.projected
    dW = param.bal_c*c_dW + param.bal_m*m_dW;
    dW = dW/size(W, 2);
else
    dW = param.bal_c*c_dW + param.bal_m*m_dW + param.lambda_W*W;
    dW = dW/size(W, 2);
end
