% Question 1
% Setting up the matrix
clear; close all; clc;
A = zeros(15, 40);
B = ones(8,2);
C = ones(2,6);
c = 2;
r = 2;
for i = 1:5
  A(r:r+7,c:c+1) = B;
  c = c+8;
  r = r+1;
end 
r = 9;
c = 10;
for i = 1:4
  A(r:r+1,c:c+5) = C;
  c = c+8;
  r = r+1;
end   
A(5:6,2:7) = C;
A(6:7,10:15) = C;
A(3:4,10:15) = C;
A(6:7,34:39) = C;
A(2:9,6:7) = B;
A(6:13,38:39) = B;
figure()
spy(A)
% Q-1(a)
S = svd(A);
figure()
plot(S);
title('Plot of singular values using Plot Function')
figure()
semilogy(S);
title('Plot of singular values using semilogy Function')
r = rank(A);
display("The exact rank of A is " + r);

% Q-1(b)
[U,S,V] = svd(A);
B = zeros(15,40);
figure
for k = 1:r
    B = B + S(k,k)*U(:,k)*V(:,k)';
    subplot(6,2,k)
    pcolor(B);
    colormap(gray);
    axis ij
    title(sprintf('Plot for Rank %d', k));
end

% The exact rank of the matrix is 10 although we see 12 non-zero singular
% values. This is because the last two very small sinular values are
% introduced by round-off errors in MATLAB. We can also see their are 10
% independent columns in the matrix A.