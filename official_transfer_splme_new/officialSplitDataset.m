function [DS_official, U_train, U_test, param_train, param_test] = officialSplitDataset(DS, U, param)



train_classes = 1:50;
test_classes = [6, 14, 15, 18, 24, 25, 34, 39, 42, 48];
train_classes(find(ismember(train_classes, test_classes))) = [];

protoStartIdx = [0; cumsum(param.numPrototypes)];



train_examples_idx = find(ismember(DS.DL, train_classes));
X_train = DS.D(:, train_examples_idx);
Y_train = DS.DL(train_examples_idx);
I_train = DS.DI(train_examples_idx);

local_train_examples_idx = find(ismember(DS.DL, test_classes));
local_X_train = DS.D(:, local_train_examples_idx);
local_Y_train = DS.DL(local_train_examples_idx);
local_I_train = DS.DI(local_train_examples_idx);

test_examples_idx = find(ismember(DS.TL, test_classes));
X_test = DS.T(:, test_examples_idx);
Y_test = DS.TL(test_examples_idx);
I_test = DS.TI(test_examples_idx);

% DS_official
DS_official.D = X_train;
DS_official.DL = Y_train;
DS_official.DI = I_train;
DS_official.LD = local_X_train;
DS_official.LDL = local_Y_train;
DS_official.LDI = local_I_train;
DS_official.T = X_test;
DS_official.TL = Y_test;
DS_official.TI = I_test;


% U_train, param_train
U_train = [];
for cls=train_classes
    trainProtoIdx = protoStartIdx(cls)+1:protoStartIdx(cls+1);
    U_train = [U_train U(:, trainProtoIdx)];
end

param_train = param;
param_train.numClasses = length(train_classes);
param_train.numPrototypes = param.numPrototypes(train_classes);
param_train.train_classes = train_classes;
param_train.knnGraphs = param.knnGraphs(train_classes);

% U_test, param_test
U_test = [];
for cls=test_classes
    testProtoIdx = protoStartIdx(cls)+1:protoStartIdx(cls+1);
    U_test = [U_test U(:, testProtoIdx)];
end

param_test = param;
param_test.numClasses = length(test_classes);
param_test.numPrototypes = param.numPrototypes(test_classes);
param_test.test_classes = test_classes;
param_test.knnGraphs = param.knnGraphs(test_classes);

prototypeRecounter = 1;
prototypeNumberMapper = [];
for cls=test_classes
    testProtoIdx = protoStartIdx(cls)+1:protoStartIdx(cls+1);
    numTestProtos = length(testProtoIdx);

    recountered = prototypeRecounter:prototypeRecounter+numTestProtos-1;
    prototypeMap = [testProtoIdx' recountered'];

    prototypeNumberMapper = [prototypeNumberMapper; prototypeMap];
    prototypeRecounter = prototypeRecounter + numTestProtos;
end

localProtoAssign_raw = param.protoAssign(local_train_examples_idx);
[~, map] = ismember(localProtoAssign_raw, prototypeNumberMapper(:, 1));
param_test.localProtoAssign = prototypeNumberMapper(map, 2);





