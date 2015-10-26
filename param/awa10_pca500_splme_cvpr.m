param.numClasses = 10;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 100;
param.lowDim = 3;
param.featureDim = 500;

param.num_clusters = 10;

% batchSize
param.c_batchSize = 100;
param.m_batchSize = 100;
param.s_batchSize = 100;

param.lambda_W = 0.001; % regularizer coefficient
param.lambda_U = 0.00001; % regularizer coefficient
param.lr_W = 1; % learning rate for W
param.lr_U = 0.01; % learning rate for U

% large margins
param.c_lm = 10000000;
param.m_lm = 100000;
param.s_lm = 100000;
param.s_sigma = 10000;

param.bal_c = 0.5;
param.bal_m = 0.1;
param.bal_s = 0.4;


