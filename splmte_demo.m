local_env = 0;
if ~local_env, cd '/v9/code/GraphTransfer/'; end

addpath 'util/'
addpath 'param/'
addpath 'sc/'
addpath 'clustering/'
addpath 'graph/'
addpath 'splmte/'
addpath 'tool/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'

% dataset = 'awa50';
% dataset = 'awa50_pca500';
dataset = 'awa10_pca500';
method = 'splmte_cvpr';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);
% param.numInstancesPerClass = hist(DS.DL, 50)';
param.numInstancesPerClass = hist(DS.DL, 10)';

splmte;







