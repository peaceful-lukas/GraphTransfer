function U = local_learnU(DS, W, U, param, trainTargetClasses, targetProtoIdx, debugMode)

if nargin < 7, debugMode = false; end
if nargin < 6, targetProtoIdx = []; end
if nargin < 5, trainTargetClasses = []; end


U_orig = U;

WX = W*DS.D;
aux = eye(sum(param.numPrototypes));

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterU;
    dU = computeGradient(DS, W, WX, U, U_orig, aux, param);
    U = update(U, dU, param);

    if debugMode
        if ~mod(n, dispCycle)
            timeElapsed = toc;
            fprintf('U%d) ', n);
            loss = local_sampleLoss(DS, W, U, U_orig, param);
            fprintf('avg time: %f\n', timeElapsed/dispCycle);

            tic;
        end
    end

    n = n + 1;
end

U_retrained = U;




% update
function U = update(U, dU, param)

U = U - param.lr_U_local * dU;


% gradient computation
function dU = computeGradient(DS, W, WX, U, U_orig, aux, param)

cTriplets = local_sampleClassificationTriplets(DS, W, U, param);
mTriplets = local_sampleMembershipTriplets(DS, W, U, param);
[sTriplets total_num_sTriplets] = local_sampleStructurePreservingTriplets(U, param);
[sPairs total_num_sPairs] = local_sampleStructurePreservingUnaryPairs(U, param);

num_cTriplets = size(cTriplets, 1);
num_mTriplets = size(mTriplets, 1);
num_sTriplets = size(sTriplets, 1);
num_sPairs = size(sPairs, 1);


c_dU = zeros(size(U));
if num_cTriplets > 0
    grad_yi = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2)))*aux(:, cTriplets(:, 2))';
    grad_c  = -2*(WX(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3)))*aux(:, cTriplets(:, 3))';
    c_dU = grad_yi - grad_c;
    % c_dU = bsxfun(@rdivide, c_dU, repelem(param.numInstancesPerClass', param.numPrototypes)); % normalize by the number of instances per each class
    c_dU = c_dU/param.c_batchSize; % normalize by the number of samples for SGD
end

m_dU = zeros(size(U));
if num_mTriplets > 0
    grad_m = -2*(WX(:, mTriplets(:, 1)) - U(:, mTriplets(:, 2)))*aux(:, mTriplets(:, 2))';
    grad_t = -2*(WX(:, mTriplets(:, 1)) - U(:, mTriplets(:, 3)))*aux(:, mTriplets(:, 3))';
    m_dU = grad_m - grad_t;
    m_dU = bsxfun(@rdivide, m_dU, repelem(param.numInstancesPerClass', param.numPrototypes)); % normalize by the number of instances per each class
    m_dU = m_dU/param.m_batchSize; % normalize by the number of samples for SGD
end

s_dU = zeros(size(U));
if num_sTriplets > 0
    s1 = -2*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3)))*aux(:, sTriplets(:, 1))';
    s2 = -2*(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 2)))*aux(:, sTriplets(:, 2))';
    s3 = -2*(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 3)))*aux(:, sTriplets(:, 3))';

    s_dU = s1 + s2 + s3;
    s_dU = s_dU/total_num_sTriplets; % normalize by the number of samples for SGD
end

sp_dU = zeros(size(U));
if num_sPairs > 0
    sp1 = -2*(U(:, sPairs(:, 1)) - U(:, sPairs(:, 2)))*aux(:, sPairs(:, 1))';
    sp2 = -2*(U(:, sPairs(:, 2)) - U(:, sPairs(:, 1)))*aux(:, sPairs(:, 2))';

    sp_dU = sp1 + sp2;
    sp_dU = sp_dU/total_num_sPairs;
end

dU = param.bal_c*c_dU + param.bal_m*m_dU + param.bal_s*(s_dU + sp_dU) + param.lambda_U_local*(U - U_orig);
dU = dU/size(U, 2);


