param.numClasses = 50;
param.numClasses = 50;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 100;
param.lowDim = 100;
param.featureDim = 500;

param.num_clusters = 10;

param.cl_same_batchSize = 100;
param.cl_diff_batchSize = 100;

param.lr_W = 0.001; % learning rate for W
param.lr_U = 1; % learning rate for U
param.lambda_W = 1000; % regularizer coefficient
param.lambda_U = 0001; % regularizer coefficient

param.cl_same_bound = 10;
param.cl_diff_bound = 100;
param.sp_near_bound = 5;
param.sp_dist_bound = 10;

param.bal_cl = 0.5;
param.bal_sp = 1-param.bal_cl;


