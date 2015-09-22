function saveResult(method, dataset, accuracy, result, local)
    if local
        save_fname = sprintf('~/Desktop/exp_results/%s/%s_%s_%d.mat', dataset, method, dataset, round(10000*accuracy));
        save(save_fname, 'result');
    else
        save_fname = sprintf('/v9/exp_results/%s/%s_%s_%d.mat', dataset, method, dataset, round(10000*accuracy));
        save(save_fname, 'result');
    end
end
