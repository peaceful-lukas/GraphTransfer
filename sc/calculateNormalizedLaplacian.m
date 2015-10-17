function NL = calculateNormalizedLaplacian(D, A)

% NL = D^(-1/2) .* L .* D^(-1/2);
NL = zeros(size(A));
for i=1:size(A,1)
    for j=1:size(A,2)
        NL(i,j) = A(i,j) / (sqrt(D(i,i)) * sqrt(D(j,j)));
    end
end