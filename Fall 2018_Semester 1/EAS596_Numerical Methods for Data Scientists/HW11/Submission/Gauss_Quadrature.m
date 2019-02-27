function [x, y] = Gauss_Quadrature(a, b)
% function to compute integral using Guassian Quadrature
x(1) = -sqrt(1/3);
x(2) = sqrt(1/3);
x = (a * (1 - x) + b * (1 + x)) / 2;
y(1) =  1;
y(2) =  y(1);
y = y * (b - a) / 2;
end