param.numClasses = 50;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 20;
param.c_batchSize = 100;
param.s_batchSize = 100;
param.lowDim = 50;
param.featureDim = 9216;

param.knn_const = 3; % constant for constructing k-nn graph.
param.c_lm = 10; % large margin for classification
param.s_lm = 0.1; % large margin for structure preserving
param.lambda_W = 0; % regularizer coefficient
param.lambda_U = 0; % regularizer coefficient
param.lr_W = 0.001; % learning rate for W
param.lr_U = 0.001; % learning rate for U
param.bal_c = 1;
param.bal_s = 0;


param.lambda_U_local = 0.1; % or 5
param.lr_U_local = 0.01;
param.bal_c = 1;
param.bal_s = 1;