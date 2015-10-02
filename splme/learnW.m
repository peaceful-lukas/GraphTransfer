function W = learnW(DS, W, U, param)

dispCycle = 100;
n = 1;

tic;
while n <= param.maxIterW
    cTriplets = sampleClassificationTriplets(DS, W, U, param);
    % cPairs = sampleClassificationUnaryPairs(DS, W, U, param);

    % dW = computeGradient(DS, W, U, cTriplets, cPairs, param);
    dW = computeGradient(DS, W, U, cTriplets, param);
    W = update(W, dW, param);

    if ~mod(n, dispCycle)
        timeElapsed = toc;
        fprintf('W%d) ', n);
        loss = sampleLoss(DS, W, U, param);
        fprintf('avg time: %f\n', timeElapsed/dispCycle);

        tic;
    end

    n = n + 1;
end


% update
function W = update(W, dW, param)

W = W - param.lr_W * dW;



% gradient computation
function dW = computeGradient(DS, W, U, cTriplets, param)

X = DS.D;
num_cTriplets = size(cTriplets, 1);
% num_cPairs = size(cPairs, 1);

c_dW = zeros(size(W));
if num_cTriplets > 0
    c_dW = (U(:, cTriplets(:, 3)) - U(:, cTriplets(:, 2))) * X(:, cTriplets(:, 1))';
    c_dW = c_dW/num_cTriplets;
end

% cp_dW = zeros(size(W));
% if num_cPairs > 0
%     cp_dW = -U(:, cPairs(:, 2))*X(:, cPairs(:, 1))';
%     cp_dW = cp_dW/num_cPairs;
% end

bal_c = param.bal_c/(param.bal_c + param.bal_s);

% dW = bal_c*c_dW + bal_c*cp_dW + param.lambda_W*W;
dW = bal_c*c_dW + param.lambda_W*W;

