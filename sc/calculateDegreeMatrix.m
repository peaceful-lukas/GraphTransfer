function D = calculateDegreeMatrix(A)

D = zeros(size(A));
for i=1:size(A,1)
    D(i,i) = sum(A(i,:));
end