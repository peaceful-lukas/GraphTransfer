function mPairs = sampleMembershipUnaryPairs(DS, W, U, param)

% (i, m)
X = DS.D;
Y = DS.DL;

i_vec = randi(numel(Y), param.m_batchSize, 1);
m_vec = param.protoAssign(i_vec);

mPairs = [i_vec, m_vec];

mpErr_vec = sum((W*X(:, mPairs(:, 1)) - U(:, mPairs(:, 2))).^2, 1) - param.m_sigma;
viol_vec = find(mpErr_vec > 0);

mPairs = mPairs(viol_vec, :);


