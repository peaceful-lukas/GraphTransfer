function A = calculateAffinityMatrix(method, X)

numExamples = size(X, 2);
A = zeros(numExamples, numExamples);

if strcmp(method, 'splme_dist')
    A = pdist(DS.D');
    A = squareform(A);
    sigma = 1;
    A = A./max(max(A));
    A = exp(-A/(2*sigma^2));
    A = A - eye(size(A));
elseif strcmp(method, 'splme_sim')
    X = normc(X);
    A = X'*X;
end

