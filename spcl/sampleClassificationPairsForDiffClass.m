function cl_diff_pairs = sampleClassificationPairsForDiffClass(DS, W, U, param)

%----------------- sampling
X = DS.D;
Y = DS.DL;
num_train_examples = length(DS.DL);

i_vec = randi(num_train_examples, param.cl_diff_batchSize, 1);
yi_vec = Y(i_vec);
k_vec = param.protoAssign(i_vec);
l_vec = ceil(param.numPrototypes(yi_vec).*rand(length(i_vec), 1));

collapsed_vec = find(k_vec == l_vec);
i_vec(collapsed_vec) = [];
l_vec(collapsed_vec) = [];

cl_diff_pairs = [i_vec, l_vec];

cl_diff_err_vec = param.cl_diff_bound - sum((W*X(:, cl_diff_pairs(:, 1)) - U(:, cl_diff_pairs(:, 2))).^2, 1);
viol_vec = find(cl_diff_err_vec > 0);

cl_diff_pairs = cl_diff_pairs(viol_vec, :);
