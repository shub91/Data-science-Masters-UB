% Question 2(b)
% Part i
clear; close all; clc;
X = [1.02;0.95;0.87;0.77;0.67;0.56;0.44;0.3;0.16;0.01];
Y = [0.39;0.32;0.27;0.22;0.18;0.15;0.13;0.12;0.13;0.15];
%s = rng;
%rand('state', 17); % to make the results repeatable
rng(17, 'v5uniform'); % to make the results repeatable
a = -0.005;
b = 0.005;
r1 = (b-a).*rand(10,1) + a; %uniformly distributed random noise btw [-0.005,0.005]
r2 = (b-a).*rand(10,1) + a;
Xn = X+r1; % Adding same noise to X and Y
Yn = Y+r2;
Cn = Q1b(Xn, Yn); % Function call to find coefficients
C = Q1b(X, Y);
fprintf('The Coefficients are: \n')
disp(Cn)
figure(6)
plot(Xn,Yn,'kx') % Plotting the noisy points
hold on
plot(X,Y,'r*') % Plotting the points
hold on
syms x y
fimplicit((Cn'*[y^2;x*y;x;y;1])-x^2) % Plotting the equation with noise
hold on
fimplicit((C'*[y^2;x*y;x;y;1])-x^2, 'b') % Plotting the original equation
axis square
axis([-1 1.5 -0.5 2]) % Setting the axis size
title('Elliptical Orbit of the planet');
legend('Data points with noise','Original Data points','Ellipse with noisy data' , 'Ellipse with original data');
%axis([-40 5 -2 60]) % Setting the axis size

% Part ii
B = zeros(10,1);
A = [Y.^2 X.*Y X Y ones(size(X))];
B = X.^2;
Bn = zeros(10,1);
An = [Yn.^2 Xn.*Yn Xn Yn ones(size(Xn))];
Bn = Xn.^2;
[U, S, V] = svd(A, 'econ');
[Un, Sn, Vn] = svd(An, 'econ');

Pn = zeros(size(An));
for i = 1:5
    r = rank(A,10^(-i));
    %rn = rank(An,10^(-i));
    P = zeros(size(A));
    %disp(r);
    %disp(rn);
    for j = 1:r
        P = P + S(j,j)*U(:,j)*V(:,j)';
        Pn = Pn + Sn(j,j)*Un(:,j)*Vn(:,j)';
    end
    Cin = Pn\Bn;
    Ci = P\B;
    figure(i)
    plot(X,Y,'r*') % Plotting the original points
    hold on
    plot(Xn,Yn,'kx') % Plotting the noisy points
    hold on
    syms x y
    fimplicit((Cin'*[y^2;x*y;x;y;1])-x^2) % Plotting the equation with noise
    hold on
    fimplicit((Ci'*[y^2;x*y;x;y;1])-x^2) % Plotting the original equation
        
    axis square
    axis([-1 1.5 -0.5 2]) % Setting the axis size
    title('Elliptical Orbits of the planet');
    legend('Original Data points','Data points with noise','Ellipse with noisy data' , 'Ellipse with original data');
    %axis([-100 100 -100 100]) % Setting the axis size
end
