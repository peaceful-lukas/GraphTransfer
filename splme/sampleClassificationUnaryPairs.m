function pairs = sampleClassificationUnaryPairs(DS, W, U, param)

i_vec = randperm(length(DS.DL), param.c_batchSize)';
yi_vec = param.protoAssign(i_vec);

useless = find(yi_vec == -1);
i_vec(useless) = [];
yi_vec(useless) = [];

pairs = [i_vec yi_vec];

loss = param.c_unary - diag((W*DS.D(:, i_vec))'*U(:, yi_vec));
valids = find(loss > 0);

pairs = pairs(valids, :);