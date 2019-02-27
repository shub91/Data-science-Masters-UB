% Question 2
clear; close; clc;
x_5 = linspace(-1, 1, 5);
x_11 = linspace(-1, 1, 11);
x_21 = linspace(-1, 1, 21);
y_5 = Runge_fn(x_5);
y_11 = Runge_fn(x_11);
y_21 = Runge_fn(x_21);

x = linspace(-1, 1, 1000);

% Q2(a)
figure;
hold on;
plot(x, Lag_fn(x, x_5, y_5),'-.r');
plot(x, Lag_fn(x, x_11, y_11),'--g');
plot(x, Lag_fn(x, x_21, y_21),'b');
plot(x, Runge_fn(x),'m');
xlabel("x");
ylabel("f(x)");
legend("LIP with 5 points", "LIP with 11 points", "LIP with 21 points", "Actual fn.","Location","eastoutside");
title("Q2(a): Actual Runge's function with Lagrange interpolanting polynomials(LIP)");
axis([-1 1 -2 2]);

% Q2(b)
x = linspace(-1, 1, 1000);
lin_5 = interp1(x_5, y_5, x, 'linear');
lin_11 = interp1(x_11, y_11, x, 'linear');
lin_21 = interp1(x_21, y_21, x, 'linear');
figure;
hold on;
plot(x, lin_5,'-.r');
plot(x, lin_11,'--g');
plot(x, lin_21,'b');
plot(x, Runge_fn(x),'m');
title("Q2(b): Linear splines and Actual Runge's function");
xlabel("x");
ylabel("f(x)");
legend("5 point", "11 point", "21 point", "Actual Fn.", "Location","eastoutside");

% Q2(c)
x = linspace(-1, 1, 1000);
% Using 'pchip' in place of 'cubic' (which is deprecated)
lin_5 = interp1(x_5, y_5, x, 'pchip');
lin_11 = interp1(x_11, y_11, x, 'pchip');
lin_21 = interp1(x_21, y_21, x, 'pchip');
figure;
hold on;
plot(x, lin_5, '-.r');
plot(x, lin_11, '--g');
plot(x, lin_21, 'b');
plot(x, Runge_fn(x),'m');
title("Q2(c): Cubic interpolants of Runge's function and Actual Runge's function");
xlabel("x");
ylabel("f(x)");
legend("5 points", "11 points", "21 points", "Actual Fn.", "Location","eastoutside");
