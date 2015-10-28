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

if param.projected
    for i=1:size(U, 2)
        normUi = norm(U(:, i), 2);
        coef = abs(1 - param.lambda_U / normUi);
        U(:, i) = coef * U(:, i);
    end
end



% gradient computation
function dU = computeGradient(DS, W, U, WX, aux, param)

X = DS.D;

cTriplets = sampleClassificationTriplets(DS, W, U, param);
% mTriplets = sampleMembershipTriplets(DS, W, U, param);
% [sTriplets total_num_sTriplets] = sampleStructurePreservingTriplets(U, param);
% [sPairs total_num_sPairs] = sampleStructurePreservingUnaryPairs(U, param);

num_cTriplets = size(cTriplets, 1);
% num_mTriplets = size(mTriplets, 1);
% num_sTriplets = size(sTriplets, 1);
% num_sPairs = size(sPairs, 1);

c_dU = zeros(size(U));
if num_cTriplets > 0
    grad_yi = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2)))*aux(:, cTriplets(:, 2))';
    grad_c  = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3)))*aux(:, cTriplets(:, 3))';
    c_dU = grad_yi - grad_c;
    % c_dU = bsxfun(@rdivide, c_dU, repelem(param.numInstancesPerClass', param.numPrototypes)); % normalize by the number of instances per each class
    c_dU = c_dU/param.c_batchSize; % normalize by the number of samples for SGD
end

% m_dU = zeros(size(U));
% if num_mTriplets > 0
%     grad_m = -2*(WX(:, mTriplets(:, 1)) - U(:, mTriplets(:, 2)))*aux(:, mTriplets(:, 2))';
%     grad_t = -2*(WX(:, mTriplets(:, 1)) - U(:, mTriplets(:, 3)))*aux(:, mTriplets(:, 3))';
%     m_dU = grad_m - grad_t;
%     m_dU = bsxfun(@rdivide, m_dU, repelem(param.numInstancesPerClass', param.numPrototypes)); % normalize by the number of instances per each class
%     m_dU = m_dU/param.m_batchSize; % normalize by the number of samples for SGD
% end

% s_dU = zeros(size(U));
% if num_sTriplets > 0
%     s1 = -2*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3)))*aux(:, sTriplets(:, 1))';
%     s2 = -2*(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 2)))*aux(:, sTriplets(:, 2))';
%     s3 = -2*(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 3)))*aux(:, sTriplets(:, 3))';

%     s_dU = s1 + s2 + s3;
%     s_dU = s_dU/total_num_sTriplets; % normalize by the number of samples for SGD
% end

% sp_dU = zeros(size(U));
% if num_sPairs > 0
%     sp1 = -2*(U(:, sPairs(:, 1)) - U(:, sPairs(:, 2)))*aux(:, sPairs(:, 1))';
%     sp2 = -2*(U(:, sPairs(:, 2)) - U(:, sPairs(:, 1)))*aux(:, sPairs(:, 2))';

%     sp_dU = sp1 + sp2;
%     sp_dU = sp_dU/total_num_sPairs;
% end

if param.projected
    dU = param.bal_c*c_dU;
    % dU = param.bal_c*c_dU + param.bal_m*m_dU + param.bal_s*(s_dU + sp_dU);
    dU = dU/size(U, 2);
else
    dU = param.bal_c*c_dU + param.bal_m*m_dU + param.bal_s*(s_dU + sp_dU) + param.lambda_U*U;
    dU = dU/size(U, 2);
end






