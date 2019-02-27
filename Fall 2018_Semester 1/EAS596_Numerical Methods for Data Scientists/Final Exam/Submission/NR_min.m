function [x_min, y_min] = NR_min(x_s, y_s, s_init, x, y)
% Function to find the coordinates of q which is closest to our given point
% p (Coordinates: (x,y). 
% Input Arguments: x_s, y_s: coordinates as function of s, on the
% parametric interface denoted by vector q, s_init: initial guess for s, x,
% y: coordinate for vector p
% Output: The coordinates of vector q which closest to vector p
syms s;
X(1,1) = s_init;
func = @(s) sqrt((x_s(s) - x)^2 + ((y_s(s) - y))^2);
error = 1; j = 1;
while error >= 10^-5
    FX(j,1) = func(X(j));
    j = j + 1;
    dfun = diff(func(s));
    d2fun = diff(dfun);
    dfun_eval = eval(subs(dfun, s, X(j - 1, 1)));
    d2f_eval = eval(subs(d2fun, s, X(j - 1, 1)));
    X(j,1) = X(j - 1, 1) - 0.5 * (dfun_eval / d2f_eval);
    FX(j, 1) = eval(subs(func, X(j)));
    error = abs(FX(j, 1) - FX(j - 1, 1));
end
x_min = x_s(X(j, 1));
y_min = y_s(X(j, 1));
end