local_env = 0;
if ~local_env, cd '/v9/code/GraphTransfer/'; end

addpath 'util/'
addpath 'param/'
addpath 'sc/'
addpath 'clustering/'
addpath 'graph/'
addpath 'splme_cvpr/'
addpath 'tool/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'

% dataset = 'awa50';
dataset = 'awa50_pca500';
method = 'splme_cvpr';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);
param.numInstancesPerClass = hist(DS.DL, 50)';

splme_cvpr;

