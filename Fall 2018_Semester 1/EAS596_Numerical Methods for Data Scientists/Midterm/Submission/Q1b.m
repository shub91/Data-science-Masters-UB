% Question 1
function C = Q1b(X, Y)
B = zeros(10,1);
A = [Y.^2 X.*Y X Y ones(size(X))];
B = X.^2;
[U, S, V] = svd(A, 'econ');
C = V*(S\U')*B;
end
