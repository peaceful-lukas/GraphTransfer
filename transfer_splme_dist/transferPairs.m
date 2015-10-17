function [tPairs str_tPairs scores S] = transferPairs(U, param)

protoStartIdx = [0; cumsum(param.numPrototypes)];
S = zeros(param.numClasses, param.numClasses); % Similarity between class prototype distributions

U_norm = normc(U);
for i=1:param.numClasses
    for j=1:param.numClasses
        if i == j
            S(i, j) = -Inf;
        else
            % maximum score
            S(i, j) = max(max(U_norm(:, protoStartIdx(i)+1:protoStartIdx(i+1))'*U_norm(:, protoStartIdx(j)+1:protoStartIdx(j+1))));

            % average score
            % S(i, j) = mean(U(:, protoStartIdx(i)+1:protoStartIdx(i+1)), 2)'*mean(U(:, protoStartIdx(j)+1:protoStartIdx(j+1)), 2);
        end
    end
end

S = triu(S);
S_sorted = sort(S(:), 'descend');

S_tmp = S;
% S_tmp(find(S_tmp < 0.7)) = 0;
S_tmp(find(S_tmp < S_sorted(param.numClasses))) = 0;

% transfer direction : <-------- ( but not important since S is a symmetric matrix.)
tPairs = [floor((find(S_tmp)-1)/param.numClasses)+1 mod(find(S_tmp), param.numClasses)];
tPairs(find(tPairs == 0)) = param.numClasses;

% Print out the transfer directions determined.
% fprintf('TRANSFER DIRECTIONS\n');
% fprintf('<------------------\n');
str_tPairs = stringifyClasses(tPairs, param.dataset);

scores = S_tmp(find(S_tmp));
[~, sorted_idx] = sort(scores, 'descend');
tPairs = tPairs(sorted_idx, :);
str_tPairs = str_tPairs(sorted_idx, :);
scores = scores(sorted_idx);


% % take it exponetially
% sim_thrsh = 0.9;
% S_exp = exp(S)./repmat(max(exp(S), [], 1), size(S, 1), 1);
% S_exp(find(S_exp < sim_thrsh)) = 0;
% % S_exp = exp(S./repmat(max(S, [], 1), size(S, 1), 1));
% % S = exp(S);

% % Best match
% [maxS, maxS_idx] = max(S, [], 1);
% tPairs = [maxS_idx' (1:param.numClasses)']; % ------> (transfer direction)


function str_tPairs = stringifyClasses(tPairs, dataset)

str_tPairs = cell(size(tPairs));

if strcmp(dataset, 'awa') || strcmp(dataset, 'AwA_30_only')
    clsname = {'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', ...
                'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', ...
                'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', ...
                'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', ...
                'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', ...
                'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', ...
                'wolf', 'chihuahua', 'rat', 'weasel', 'otter', ...
                'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', ...
                'pig', 'lion', 'mouse', 'polar+bear', 'collie',  ...
                'walrus', 'raccoon', 'cow', 'dolphin'};
    
    for i=1:size(tPairs, 1)
        str_tPairs{i, 1} = clsname{tPairs(i, 1)};
        str_tPairs{i, 2} = clsname{tPairs(i, 2)};
    end
elseif strcmp(dataset, 'pascal3d_pascal') || strcmp(dataset, 'pascal3d_imagenet') || strcmp(dataset, 'pascal3d_all')
    clsnames = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor'};
    
    for i=1:size(tPairs, 1)
        for j=1:2
            str_tPairs{i, j} = clsnames{tPairs(i, j)};
        end
    end
elseif strcmp(dataset, 'voc')
    clsnames = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', ...
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'};

    for i=1:size(tPairs, 1)
        for j=1:2
            str_tPairs{i, j} = clsnames{tPairs(i, j)};
        end
    end
else
    fprintf('\nno class name list on %s\n\n', dataset);
end







% graph Laplacian between classes
% L = {};
% for i=1:param.numClasses
%     L{i} = laplacian(param.knnGraphs{i}, 1); % normalized
% end

% M = zeros(param.numClasses, param.numClasses); % prematched score matrix by graph laplacians
% for i=1:param.numClasses
%     for j=1:param.numClasses
%         M(i, j) = prematching(L{i}, L{j});
%     end
% end

