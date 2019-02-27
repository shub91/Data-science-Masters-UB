A = [1,2,3;4,5,6;7,8,7;4,2,3;4,2,2];
[Q1, R1] = Q1CGS(A);
[Q2, R2] = Q1MGS(A);
[Q3, R3] = Q1HR(A);

cond(A)

Q1'*Q1
Q2'*Q2
Q3'*Q3

% COMMENTS
% All three algorithms provide matching results with respect to Q and R.
% We check the property of producing orthogonality(Q'Q) which also comes 
% out to be identity for all the cases since it is a well-conditioned
% matrix (condition number = 12.4508). Usually, if the matrix is not well
% conditioned classical Gram-Schmidt provide poor orthogonality, modified
% GS also behaves similarly while Householder Reflectors always give good
% orthogonality (and hence the preferred method).