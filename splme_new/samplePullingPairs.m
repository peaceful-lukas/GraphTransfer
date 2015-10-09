function pPairs = samplePullingPairs(DS, W, U, param)

% (i, k)
% i - example index
% k - corresponding prototype for each i-th example

X = DS.D;
protoStartIdx = [0; cumsum(param.numPrototypes)];

i_vec = randi(numel(DS.DL), param.p_batchSize, 1);
k_vec = param.protoAssign(i_vec);

pPairs = [i_vec k_vec];

loss_vec = sum((W*X(:, i_vec) - U(:, k_vec)).^2, 1) - param.p_sigma;
valids = find(loss_vec > 0);
pPairs = pPairs(valids, :);
