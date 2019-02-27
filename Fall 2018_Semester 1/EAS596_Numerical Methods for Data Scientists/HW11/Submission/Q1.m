% Question 1 (a) : Least Square fit
clear; close all; clc;
X = [0.1; 0.2; 0.4; 0.6; 0.9; 1.3; 1.5; 1.7; 1.8];
Y = [0.75; 1.25; 1.45; 1.25; 0.85; 0.55; 0.35; 0.28; 0.18];
B = zeros(size(X));
A = [ones(size(X)) X log(X)];
B = log(Y);
[U, S, V] = svd(A, 'econ');
C1 = V*(S\U')*B;
fprintf('The Coefficients are (log(a0) & a1) from least squares fit: \n')
disp(C1)
figure
plot(X,Y,'ko') % Plotting the given points
hold on
syms x y % Requires symbolic math toolbox
%axis square
fimplicit(((C1'*[1;x;log(x)])-log(y)), 'b'); % Plotting the equation
title('Q1(a): Linear Least Square Fit');
legend('Original Data points','Linear least square fit');
% axis([-40 5 -2 60]) % Setting the axis size
axis([-0.5 5 0 1.5]) % Setting the axis size


% Question 1(b) : Newton Raphson

tol = 1.0e-10;
a0 = 1; a1 = 1;
A = [a0;a1];
alpha = 0.1;
Y_hat = A(1) * X .* exp(A(2)*X);
R = Y - Y_hat;
S = (norm(R))^2;
SA2(1) = S;
SA2(2) = S+1;
RA2(1) = norm(R);
RA2(2) = norm(R)+1;
i = 2;
while(RA2(i)/RA2(i-1)>tol)
    i = i+1;
    Y_hat = A(1) * X .* exp(A(2)*X);
    R = Y - Y_hat;
    RA2(i) = norm(R);
    S = (norm(R))^2;
    SA2(i) = S;
    if (SA2(i)>SA2(i-1))
        break
    end
    dRda0 = -X.*exp(A(2)*X);
    dRda1 = -A(1)*(X.^2).*exp(A(2)*X);
    d2Ra1a0 = -(X.^2).*exp(A(2)*X);
    d2Ra12 = -A(1)*(X.^3).*exp(A(2)*X);
    g0 = sum(-2*R.*dRda0);
    g1 = sum(-2*R.*dRda1);
    g = [g0;g1];
    % Defining the Hessian
    H(1,1) = 2*sum(dRda0.^2);
    H(2,1) = 2*sum(dRda0.*dRda1+R.*d2Ra1a0);
    H(1,2) = H(2,1);
    H(2,2) = 2*sum(dRda1.^2+R.*d2Ra12);
    J = [dRda0 dRda1];
    %display(A);
    %display(H);
    D = (-1)*H\g;
    A = A + alpha * D;
    alpha = alpha + 0.0001; % line search
   % so = S;
end
fprintf('The Coefficients are (a0 and a1) from Newton Raphson method: \n')
disp(A)
fprintf('The Objective function values for each iteration for NR method: \n')
disp(SA2(3:end))
fprintf('The norm(r) for each iteration for NR method: \n')
disp(RA2(3:end))
figure
plot(X,Y,'ko') % Plotting the given points
hold on
syms x y % Requires symbolic math toolbox
%axis square
fimplicit(y-A(1)*x*exp(A(2)*x), 'b'); % Plotting the equation
title('Q1(b): Newton Raphson (NR) method');
legend('Original Data points','Min. residual fit using NR method');
axis([-0.5 5 0 1.5]) % Setting the axis size

% Question 1(c): Guass Newton

tol = 1.0e-10;
a0 = 1; a1 = 1;
A = [a0;a1];
alpha3 = 0.1;
Y_hat = A(1) * X .* exp(A(2)*X);
R = Y - Y_hat;
S = (norm(R))^2;
SA1(1) = S;
SA1(2) = S+1;
RA1(1) = norm(R);
RA1(2) = norm(R)+1;
i=2;
while(RA1(i)/RA1(i-1)>tol)
    i = i+1;
    Y_hat = A(1) * X .* exp(A(2)*X);
    R = Y - Y_hat;
    RA1(i) = norm(R);
    S = (norm(R))^2;
    SA1(i) = S;
    if (SA1(i)>SA1(i-1))
        break
    end
    dRda0 = -X.*exp(A(2)*X);
    dRda1 = -A(1)*(X.^2).*exp(A(2)*X);
    J = [dRda0 dRda1];
    %display(A);
    %display(J);
    D = (-1)*(J'*J)\(J'*R);
    A = A + alpha3 * D;
    alpha3 = alpha3 + 0.1; % line search
    %so = S;
    %Ro=R;
    %i = i+1;
end
fprintf('The Coefficients are (a0 and a1) from GN method: \n')
disp(A)
fprintf('The Objective function values for each iteration for GN method: \n')
disp(SA1(3:end))
fprintf('The norm(r) for each iteration for GN method: \n')
disp(RA1(3:end))
figure
plot(X,Y,'ko') % Plotting the given points
hold on
syms x y % Requires symbolic math toolbox
%axis square
fimplicit(y-A(1)*x*exp(A(2)*x), 'b'); % Plotting the equation
title('Q1(c): Guass Newton Method(GN)');
legend('Original Data points','Min. residual fit using GN method');
axis([-0.5 5 0 1.5]) % Setting the axis size

% We observe that the coefficients obtained for linear least square
% analysis and Gauss-Newton method agree with each other however they do
% not agree with Newton-Raphson method. For both, Gauss-Newton as well as
% Newton-Raphson methods line search was performed with an initial alpha
% value of 0.1 and updating it by 0.1 at each iteration. For Newton-Raphson
% method we see that the norm(r) i.e. the residuals increases at 1st
% iteration itself and the objective function value also increases whereas we get
% convergence for Gauss-Newton. This is because we take the second
% derivative term in the Hessian in Newton Raphson method which are noisy.
    