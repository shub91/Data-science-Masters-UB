% Question 2 (d)
clc
clear all
close all

% Loading the time series data: theta_1
d = load('theta1.mat');
theta_mat = d.t_theta;
L_1 = 0.5;
L_2 = 0.75;
h = 0;

% c1 = 6.2185   c2 = 0.9826
init = [6.2185, 0.9826];
theta_mat(2,:) = (init(1).*theta_mat(1,:)./(theta_mat(1,:)+init(2)));
% Part(b):ii
for i = 1:20
    theta1 = theta_mat(2,i);
%   f = inline('[x(1) - 0.5*cos('+ th +') + 0.75*cos(x(2)); 0 - 0.5*sin(' + + ') + 0.75*sin(x(2))]', 'x');
    J = inline('[1, 0.75*sin(x(2)); 0, (-1)*0.75*cos(x(2))]', 'x' );
    f = @(x) [x(1) - L_1*cos(theta1) - L_2*cos(x(2)); h - L_1*sin(theta1) - L_2*sin(x(2))];

    % Initial guess
    x0 = [1; 0];

    % Solver tolerance
    tol = 1.0e-10;

    % Max number of Newton iterations
    max_iters = 5000;

    % Run our Newton code
    [ x, error_est, n ] = newton_sys(f,J,x0, tol, max_iters );
    X(:,i) = x;
end
% Part(b): iii
figure;
hold on;
title("Q2d - Plot of \theta_1, \theta_2, and x vs t");
xlabel("Time (t)");
ylabel("Variables (\theta_1, \theta_2, x)");
axis tight;
plot(theta_mat(1,:), theta_mat(2, :), 'b');
plot(theta_mat(1,:), X(2, :), 'g-');
plot(theta_mat(1,:), X(1, :), 'k-.');
legend('\theta_1', '\theta_2', 'x(t)', 'Location', "eastoutside");

% Part(b): iv

vel = der(X(1,:),(theta_mat(1,:)));
acc = der(vel,(theta_mat(1,:)));

% Part(b): v

figure;
hold on;
title("Q2d:  Plot for Computed piston force vs. x");
xlabel("x");
ylabel("Computed Piston Force");
axis square;
plot(X(1,:), 10 * acc, 'r-');

% Part(b): vi
fx = @(init,t)(init(1).*t)./(t+init(2));
w = 10 * trapz(X(1, :), acc);

fprintf('The work done by the piston is (using the fitted curve): %f \n',w);  


