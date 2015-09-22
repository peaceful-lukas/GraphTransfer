% cd /v9/code/GraphTransfer

addpath 'util'
addpath 'param'
addpath 'ddcrp'
addpath 'blme'
addpath 'pgm'
addpath 'pgm/RRWM'
addpath 'transfer'

dataset = 'pascal3d_pascal';
method = 'blme';

DS = loadDataset(dataset, 1);
param = setParameters(dataset, method);

[W U] = blme(DS, param);
