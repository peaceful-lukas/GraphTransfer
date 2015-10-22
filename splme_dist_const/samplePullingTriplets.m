function pTriplets = samplePullingTriplets(DS, W, U, param)

% (i, k, l)
% i - example index
% k - corresponding prototype for each i-th example
% l - the index of the other class prototypes for x_i

X = DS.D;
offset_vec = [0; cumsum(param.numPrototypes)];

i_vec = randi(numel(DS.DL), param.p_batchSize, 1);
k_vec = param.protoAssign(i_vec);
yi_vec = DS.DL(i_vec);
l_vec = offset_vec(yi_vec) + randi(param.num_clusters, length(yi_vec), 1);
collapsed = find(l_vec == k_vec);

i_vec(collapsed) = [];
k_vec(collapsed) = [];
l_vec(collapsed) = [];

pTriplets = [i_vec k_vec l_vec];


loss_vec = param.p_lm + sum((W*X(:, pTriplets(:, 1)) - U(:, pTriplets(:, 2))).^2, 1) - sum((W*X(:, pTriplets(:, 1)) - U(:, pTriplets(:, 3))).^2, 1);
valids = find(loss_vec > 0);
pTriplets = pTriplets(valids, :);

