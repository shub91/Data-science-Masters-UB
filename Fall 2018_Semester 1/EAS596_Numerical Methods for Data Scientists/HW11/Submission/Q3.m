% Question 3
clear; close; clc;

a = 1;
b = 2;
f = @(x) ((1 + x.^ -2)).^ 0.5;
analytical = integral(f, a, b);

% Q3(a)

sTrap = 1;
int_trap = Trapez(a, b, sTrap, f);
while(abs((analytical - int_trap) / analytical) > 0.001)
    sTrap = sTrap + 1;
    int_trap = Trapez(a, b, sTrap, f);
end
disp("Integral value using Trapezoidal method: " + int_trap);

% Q3(b)

sSimp = 1;
int_simp = Simpson(a, b, sSimp, f);
while(abs((analytical - int_simp) / analytical) > 0.001)
    sSimp = sSimp + 1;
    int_simp = Simpson(a, b, sSimp, f);
end
disp("Integral value using Simpson's method: " + int_simp);
% Q3(c)

[u, v] = Gauss_Quadrature(a, b);
fVals = f(u);
int_GQ = dot(v, fVals);
err = abs((analytical - int_GQ) / analytical);

disp("Integral value using 2-point Gauss Quadrature method: " + int_GQ);

disp(" ");
disp("No. of intervals required for Trapezoidal method: " + sTrap);
disp("No. of intervals required for Simpson method: " + sSimp);
