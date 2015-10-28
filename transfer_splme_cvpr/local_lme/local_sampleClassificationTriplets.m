function cTriplets = sampleClassificationTriplets(DS, W, U, param)

% (i, yi, c)
X = DS.D;
Y = DS.DL;
offset_vec = [0; cumsum(param.numPrototypes)];

i_vec = randi(numel(Y), param.c_batchSize, 1);
yi_vec = ceil(param.numPrototypes(Y(i_vec)).*rand(length(i_vec), 1));
yi_vec = yi_vec + offset_vec(Y(i_vec));

class_vec = randi(param.numClasses, length(i_vec), 1);
collapsed = find(Y(i_vec) == class_vec);

i_vec(collapsed) = [];
yi_vec(collapsed) = [];
class_vec(collapsed) = []; % drop

c_vec = ceil(param.numPrototypes(class_vec).*rand(length(i_vec), 1));
c_vec = c_vec + offset_vec(class_vec);

cTriplets = [i_vec yi_vec c_vec];

c_err_vec = param.c_lm + sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 2))).^2, 1) - sum((W*X(:, cTriplets(:, 1)) - U(:, cTriplets(:, 3))).^2, 1);
viol_vec = find(c_err_vec > 0);


cTriplets = cTriplets(viol_vec, :);
