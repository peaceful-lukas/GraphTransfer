local_env = 1;

addpath 'util/'
addpath 'param/'
addpath 'blme_sim/'
addpath 'tool/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'

dataset = 'voc';
method = 'blme_sim';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U] = blme_sim(DS, param);
blme_sim;
