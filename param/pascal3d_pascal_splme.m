param.numClasses = 12;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 50;
param.c_batchSize = 100;
param.s_batchSize = 100;
param.lowDim = 12;
param.featureDim = 9216;

param.knn_const = 3; % constant for constructing k-nn graph.
param.c_lm = 10; % large margin for classification
param.s_lm = 0.1; % large margin for structure preserving
param.lambda_W = 1; % regularizer coefficient
param.lambda_U = 0.1; % regularizer coefficient
param.lr_W = 0.0001; % learning rate for W
param.lr_U = 0.00001; % learning rate for U
param.bal_c = 1;
param.bal_s = 10;