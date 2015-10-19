function loss = sampleLoss(DS, W, U, param)

X = DS.D;
mTriplets = sampleMembershipTriplets(DS, W, U, param);
pPairs = samplePullingPairs(DS, W, U, param);

num_mTriplets = size(mTriplets, 1);
num_pPairs = size(pPairs, 1);


mErr = 0;
num_mV = 0;
if num_mTriplets > 0
    mErr_vec = param.m_lm + diag((W*X(:, mTriplets(:, 1)))' * (U(:, mTriplets(:, 3)) - U(:, mTriplets(:, 2))));
    viol = find(mErr_vec > 0);
    num_mV = length(viol);

    if viol > 0
        mErr = sum(mErr_vec(viol));
    end
end

pErr = 0;
num_pV = 0;
if num_pPairs > 0
    pErr_vec = diag((W*X(:, pPairs(:, 1)))'*U(:, pPairs(:, 2))) - param.p_sigma;
    viol = find(pErr_vec > 0);
    num_pV = length(viol);
    
    if viol > 0
        pErr = sum(pErr_vec(viol));
    end
end

bal_m = param.bal_m/(param.bal_m + param.bal_p);
bal_p = param.bal_p/(param.bal_m + param.bal_p);

mErr = bal_m*mErr;
pErr = bal_p*pErr;

loss = param.bal_m*mErr + param.bal_p*pErr + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
fprintf('mV: %d / pV: %d / mE: %f / pE: %f / normW: %f / normU: %f / ', num_mV, num_pV, mErr, pErr, norm(W, 'fro'), norm(U, 'fro'));