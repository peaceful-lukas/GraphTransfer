function sTriplets = vaildStructurePreservingTriplets(U, param)

sTriplets = param.sTriplets;

loss_vec = param.s_lm + sum(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 2)).^2) - sum(U(:, sTriplets(:, 1)) - U(:, sTriplets(:, 3)).^2);
valids = find(loss_vec > 0);
sTriplets = sTriplets(valids, :);