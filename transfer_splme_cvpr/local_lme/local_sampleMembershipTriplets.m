function mTriplets = sampleMembershipTriplets(DS, W, U, param)

% (i, m, t)
X = DS.D;
Y = DS.DL;
proto_offset = [0; cumsum(param.numPrototypes)];

i_vec = randi(numel(Y), param.m_batchSize, 1);
m_vec = param.protoAssign(i_vec);
t_vec = ceil(param.numPrototypes(Y(i_vec)).*rand(length(i_vec), 1));
t_vec = t_vec + proto_offset(Y(i_vec));

collapsed = find(m_vec == t_vec);

i_vec(collapsed) = [];
m_vec(collapsed) = [];
t_vec(collapsed) = [];

mTriplets = [i_vec, m_vec, t_vec];

mErr_vec = param.m_lm + sum((W*X(:, mTriplets(:, 1)) - U(:, mTriplets(:, 2))).^2, 1) - sum((W*X(:, mTriplets(:, 1)) - U(:, mTriplets(:, 3))).^2, 1);
viol_vec = find(mErr_vec > 0);

mTriplets = mTriplets(viol_vec, :);


