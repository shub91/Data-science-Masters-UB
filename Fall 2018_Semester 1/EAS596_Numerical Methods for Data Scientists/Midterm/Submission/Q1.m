% MATLAB MIDTERM EXAM
% Question 1
clear; close all; clc;
X = [1.02;0.95;0.87;0.77;0.67;0.56;0.44;0.3;0.16;0.01];
Y = [0.39;0.32;0.27;0.22;0.18;0.15;0.13;0.12;0.13;0.15];
C = Q1b(X, Y); % Function call to find coefficients
fprintf('The Coefficients are: \n')
disp(C)
figure
plot(X,Y,'kx') % Plotting the given points
hold on
syms x y % Requires symbolic math toolbox
axis square
fimplicit((C'*[y^2;x*y;x;y;1])-x^2, 'b') % Plotting the equation
title('Elliptical Orbit of the planet');
legend('Original Data points','Ellipse with original data');
%axis([-40 5 -2 60]) % Setting the axis size
axis([-0.5 1.5 -0.5 1.5]) % Setting the axis size
