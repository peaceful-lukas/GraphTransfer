function U = officialLearnU(DS, W, U, param, debugMode)

if nargin < 5, debugMode = false; end

U_orig = U;

WX = W*DS.LD;
aux = eye(sum(param.numPrototypes));

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterU;
    cTriplets = local_sampleClassificationTriplets(DS, W, U, param);
    pPairs = local_samplePullingPairs(DS, W, U, param);

    dU = computeGradient(WX, U, U_orig, aux, cTriplets, pPairs, param);
    U = update(U, dU, param);

    if debugMode
        if ~mod(n, dispCycle)
            timeElapsed = toc;
            fprintf('U%d) ', n);
            loss = local_sampleLoss(DS, W, U, W, U_orig, param);
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
function dU = computeGradient(WX, U, U_orig, aux, cTriplets, pPairs, param)

num_cTriplets = size(cTriplets, 1);
num_pPairs = size(pPairs, 1);

c_dU = zeros(size(U));
if num_cTriplets > 0
    c_dU = WX(:, cTriplets(:, 1))*(aux(:, cTriplets(:, 3)) - aux(:, cTriplets(:, 2)))';
    % c_dU = c_dU/param.c_batchSize;
    c_dU = c_dU/norm(c_dU, 'fro');
end


% p_dU = zeros(size(U));
% if num_pPairs > 0
%     p_dU = (U(:, pPairs(:, 2)) - WX(:, pPairs(:, 1)))*aux(:, pPairs(:, 2))';
%     % p_dU = 2*p_dU/param.p_batchSize;
%     p_dU = p_dU/norm(p_dU, 'fro');
% end

% bal_c = param.bal_c/(param.bal_c + param.bal_p + param.lambda_U_local);
% bal_p = param.bal_p/(param.bal_c + param.bal_p + param.lambda_U_local);
% lambda_U_local = param.lambda_U_local/(param.bal_c + param.bal_p + param.lambda_U_local);

% dU = bal_c*c_dU + bal_p*p_dU + lambda_U_local*(U - U_orig);



bal_c = param.bal_c/(param.bal_c + param.lambda_U_local);
lambda_U_local = param.lambda_U_local/(param.bal_c + param.lambda_U_local);

dU = bal_c*c_dU + lambda_U_local*(U - U_orig);
