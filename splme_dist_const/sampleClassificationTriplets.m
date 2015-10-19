function cTriplets = sampleClassificationTriplets(DS, W, U, param)

% (i, y_i, c)
X = DS.D;
offset_vec = [0; cumsum(param.numPrototypes)];

i_vec = randi(numel(DS.DL), param.c_batchSize, 1);
% yi_vec = param.protoAssign(i_vec);
yi_class_vec = DS.DL(i_vec);

yi_vec = offset_vec(yi_class_vec) + randi(param.num_clusters, length(yi_class_vec), 1);


class_vec = randi(param.numClasses, length(i_vec), 1);
collapsed = find(DS.DL(i_vec) == class_vec);

i_vec(collapsed) = [];
yi_vec(collapsed) = [];
class_vec(collapsed) = []; % drop

c_vec = ceil(param.numPrototypes(class_vec).*rand(length(i_vec), 1));

c_vec = c_vec + offset_vec(class_vec);

useless = find(yi_vec == -1);
i_vec(useless) = [];
yi_vec(useless) = [];
c_vec(useless) = [];

cTriplets = [i_vec yi_vec c_vec];

dist_yi_vec = sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2))).^2, 1);
dist_c_vec = sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3))).^2, 1);
loss_vec = param.c_lm + dist_yi_vec - dist_c_vec;
valids = find(loss_vec > 0);
cTriplets = cTriplets(valids, :);

