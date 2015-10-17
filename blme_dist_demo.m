local_env = 1;

addpath 'util/'
addpath 'param/'
addpath 'blme_dist/'
addpath 'tool/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'

dataset = 'voc';
method = 'blme_dist';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

% [W U] = blme_dist(DS, param);
blme_dist;
