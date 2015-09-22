function param = setParameters(dataset, method)

exec_fname = sprintf('%s_%s', dataset, method);
eval(exec_fname);

param.method = method;
param.dataset = dataset;
