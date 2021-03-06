function loss = sampleLoss(DS, W, U, U_orig, param)

X = DS.D;

cTriplets = local_sampleClassificationTriplets(DS, W, U, param);
mTriplets = local_sampleMembershipTriplets(DS, W, U, param);
[sTriplets total_num_sTriplets] = local_sampleStructurePreservingTriplets(U, param);
[sPairs total_num_sPairs] = local_sampleStructurePreservingUnaryPairs(U, param);

num_cTriplets = size(cTriplets, 1);
num_mTriplets = size(mTriplets, 1);
num_sTriplets = size(sTriplets, 1);
num_sPairs = size(sPairs, 1);


cErr = 0;
num_cV = 0;
if num_cTriplets > 0
    cErr_vec = param.c_lm + sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2))).^2, 1) - sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3))).^2, 1);
    viol_vec = find(cErr_vec > 0);
    num_cV = length(viol_vec);
    cErr = sum(cErr_vec);
    cErr = cErr/param.c_batchSize;
end


mErr = 0;
num_mV = 0;
if num_mTriplets > 0
    mErr_vec = param.m_lm + sum((W*X(:, mTriplets(:, 1)) - U(:, mTriplets(:, 2))).^2, 1) - sum((W*X(:, mTriplets(:, 1)) - U(:, mTriplets(:, 3))).^2, 1);
    viol_vec = find(mErr_vec > 0);
    num_mV = length(viol_vec);
    mErr = sum(mErr_vec);
    mErr = mErr/param.m_batchSize;
end


sErr = 0;
num_sV = 0;
if num_sTriplets > 0
    sErr_vec = param.s_lm - 2*diag(U(:, sTriplets(:, 1))'*(U(:, sTriplets(:, 2)) - U(:, sTriplets(:, 3))))    ...
            + diag(U(:, sTriplets(:, 2))'*U(:, sTriplets(:, 2)))                                   ...
            - diag(U(:, sTriplets(:, 3))'*U(:, sTriplets(:, 3)));

    viol_vec = find(sErr_vec > 0);
    num_sV = length(viol_vec);
    sErr = sum(sErr_vec);
    sErr = sErr/total_num_sTriplets;
end

spErr = 0;
num_spV = 0;
if num_sPairs > 0
    spErr_vec = param.s_sigma - sum((U(:, sPairs(:, 1)) - U(:, sPairs(:, 2))).^2, 1);
    viol_vec = find(spErr_vec > 0);
    num_spV = length(viol_vec);
    spErr = sum(spErr_vec);
    spErr = spErr/total_num_sPairs;
end

loss = param.bal_c*cErr + param.bal_m*mErr + param.bal_s*(sErr + spErr) + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U_local*0.5*norm(U-U_orig, 'fro')^2;
fprintf('cV: %d / mV: %d / sV(spV): %d(%d) / cE: %f / mE: %f / sE: %f / normU: %f / ', num_cV, num_mV, num_sV, num_spV, cErr, mErr, sErr+spErr, norm(U-U_orig, 'fro')/size(U, 2));

% loss = param.bal_c*cErr + param.bal_m*mErr + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
% fprintf('cV: %d / mV: %d / cE: %f / mE: %f / normW: %f / normU: %f / ', num_cV, num_mV, cErr, mErr, norm(W, 'fro')/size(W, 2), norm(U, 'fro')/size(U, 2));

% loss = param.bal_c*cErr + param.lambda_W*0.5*norm(W, 'fro')^2 + param.lambda_U*0.5*norm(U, 'fro')^2;
% fprintf('cV: %d / cE: %f / normW: %f / normU: %f / ', num_cV, cErr, norm(W, 'fro')/size(W, 2), norm(U, 'fro')/size(U, 2));





