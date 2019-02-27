% Question 2(a) 
% Part i
clear; close all; clc;
A = [1/3,1/3,2/3;2/3,2/3,4/3;1/3,2/3,3/3;2/5,2/5,4/5;3/5,1/5,4/5];
fprintf('The rank of the matrix A is: \n')
disp(rank(A)) %Q2(a)-i
%A = [1/sym(3),1/sym(3),2/sym(3);2/sym(3),2/3,4/3;1/3,2/3,3/3;2/5,2/5,4/5;3/5,1/5,4/5];
%rank(A);
% Part ii
[U, S, V] = svd(A);
disp(U);
disp(S);
disp(V);
fprintf('The third sigular value: \n')
fprintf('%d',S(3,3));
fprintf('\nTolerance to calculate rank: \n')
disp(max(size(A))*eps(norm(A)));
% Taking values upto 4 decimal points
Af = [0.3333 0.3333 0.6667;0.6667 0.6667 1.3333;0.3333 0.6667 1.0000; 0.4000 0.4000 0.8000; 0.6000 0.2000 0.8000];
disp(rank(Af));
[Uf, Sf, Vf] = svd(Af);
disp(Uf);
disp(Sf);
disp(Vf);
fprintf('The third sigular value: \n')
fprintf('%d',Sf(3,3));
fprintf('\nTolerance to calculate rank: \n')
disp(max(size(Af))*eps(norm(Af)));
disp(rank(Af, 1e-04)); % rank = 2
% Part iii
S = zeros(2000,2000)
for i = 1:2000
    S(i,i) = 0.9^i;
end
fprintf('Rank: \n') % 276
disp(rank(S));
fprintf('Rank on increasing the significant figures: \n')
disp(rank(vpa(S))); % 700
