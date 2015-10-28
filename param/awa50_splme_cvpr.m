param.numClasses = 50;
param.maxIterW = 1000;
param.maxIterU = 1000;
param.maxAlter = 100;
param.lowDim = 450;
param.featureDim = 9216;

param.num_clusters = 10;

param.projected = true;

% batchSize
param.c_batchSize = 100;
param.m_batchSize = 100;
param.s_batchSize = 100;

param.lambda_W = 0.02; % regularizer coefficient
param.lambda_U = 200; % regularizer coefficient
param.lr_W = 0.01; % learning rate for W
param.lr_U = 0.01; % learning rate for U

% large margins
param.c_lm = 10;
param.m_lm = 100;
% param.m_sigma = 10;
param.s_lm = 0.01;
param.s_sigma = 0.1;

param.bal_c = 0.5;
param.bal_m = 0.2;
param.bal_s = 0.3;


