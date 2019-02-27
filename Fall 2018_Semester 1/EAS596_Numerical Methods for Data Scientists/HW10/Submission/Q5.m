% Q5

syms x;
F1 = @(x) x(1)^3 +x(2)-1;
F2 = @(x) x(1)+x(2);
jcbn = @(x) [3*x(1)^2 1;
                 1 1];
             tol = 10^-6;
imax = 100000;
ig = [1 0];
f = @handle;
[root1, ic1] = N5(f,jcbn,ig',imax,tol);
ig = [1/sqrt(3) 1];
[root2, ic3] = N5(f,jcbn,ig',imax,tol);