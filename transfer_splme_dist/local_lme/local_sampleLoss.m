function loss = local_sampleLoss(DS, W, U, W_orig, U_orig, param)


X = DS.D;
cTriplets = sampleClassificationTriplets(DS, W, U, param);
pPairs = samplePullingPairs(DS, W, U, param);

num_cTriplets = size(cTriplets, 1);
num_pPairs = size(pPairs, 1);

cErr = 0;
num_cV = 0;
if num_cTriplets > 0
    cErr_vec = param.c_lm + diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))));
    viol = find(cErr_vec > 0);
    num_cV = length(viol);

    if viol > 0
        cErr = sum(cErr_vec(viol));
    end
end


pErr = 0;
num_pV = 0;
if num_pPairs > 0
    pErr_vec = sum((W*X(:, pPairs(:, 1)) - U(:, pPairs(:, 2))).^2, 1) - param.p_sigma;
    viol = find(pErr_vec > 0);
    num_pV = length(viol);

    if viol > 0
        pErr = sum(pErr_vec(viol));
    end
end

bal_c = param.bal_c/(param.bal_c + param.bal_p);
bal_p = param.bal_p/(param.bal_c + param.bal_p);

cErr = bal_c*cErr;
pErr = bal_p*pErr;


loss = param.bal_c*cErr + param.bal_p*pErr + param.lambda_U_local*0.5*norm(U - U_orig, 'fro')^2;
fprintf('cV: %d / pV: %d / cE: %f / pE: %f / norm(U - U_orig)_F: %f / ', num_cV, num_pV, cErr, pErr, norm(U-U_orig, 'fro'));
