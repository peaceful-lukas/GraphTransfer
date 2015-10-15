local_env = 1;

addpath 'util'
addpath 'param'
addpath 'blme'

dataset = 'voc';
method = 'blme';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

[W U] = blme(DS, param);
blme;
