function cTriplets = local_sampleClassificationTriplets(DS, W, U, param)

% (i, y_i, c)
X = DS.LD;

i_vec = randi(numel(DS.LDL), param.c_batchSize, 1);
yi_vec = param.localProtoAssign(i_vec);

class_vec = randi(param.numClasses, length(i_vec), 1);
collapsed = find(DS.LDL(i_vec) == class_vec);

i_vec(collapsed) = [];
yi_vec(collapsed) = [];
class_vec(collapsed) = []; % drop

offset_vec = [0; cumsum(param.numPrototypes)];
c_vec = ceil(param.numPrototypes(class_vec).*rand(length(i_vec), 1));

c_vec = c_vec + offset_vec(class_vec);


cTriplets = [i_vec yi_vec c_vec];

loss_vec = param.c_lm + diag((W*X(:, cTriplets(:, 1)))' * (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))));
valids = find(loss_vec > 0);
cTriplets = cTriplets(valids, :);

