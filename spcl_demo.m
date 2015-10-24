local_env = 1;
if ~local_env, cd '/v9/code/GraphTransfer/'; end

addpath 'util/'
addpath 'param/'
addpath 'sc/'
addpath 'clustering/'
addpath 'graph/'
addpath 'spcl/'
addpath 'tool/'
addpath 'tool/vis/'
addpath 'tool/vis/distinguishable_colors/'

% dataset = 'awa50';
dataset = 'awa50_pca500';
method = 'spcl';

DS = loadDataset(dataset, local_env);
param = setParameters(dataset, method);

spcl;




