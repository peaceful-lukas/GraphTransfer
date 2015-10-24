function cl_same_pairs = sampleClassificationPairsForSameClass(DS, W, U, param)

%----------------- sampling
X = DS.D;
num_train_examples = length(DS.DL);

i_vec = randi(num_train_examples, param.cl_same_batchSize, 1);
k_vec = param.protoAssign(i_vec);

cl_same_pairs = [i_vec, k_vec];

cl_same_err_vec = sum((W*X(:, cl_same_pairs(:, 1)) - U(:, cl_same_pairs(:, 2))).^2, 1) - param.cl_same_bound;
viol_vec = find(cl_same_err_vec > 0);

cl_same_pairs = cl_same_pairs(viol_vec, :);


%----------------- all possible pairs
% X = DS.D;

% i_vec = 1:length(DS.DL);
% k_vec = param.protoAssign(i_vec);
% cl_same_pairs = [i_vec, k_vec];

% cl_same_err_vec = sum((W*X(:, cl_same_pairs(:, 1)) - U(:, cl_same_pairs(:, 2))).^2, 1) - param.cl_same_bound;
% viol_vec = find(cl_same_err_vec > 0);

% cl_same_pairs = cl_same_pairs(viol_vec, :);

