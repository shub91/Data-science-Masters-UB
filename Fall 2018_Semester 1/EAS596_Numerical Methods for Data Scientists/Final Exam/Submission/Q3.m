% Question 3
close; 
clear; 
clc;

% Part (d)
% p = (4,2)
[x, fval] = fminbnd(@(s) sqrt((cos(s) - 4)^2 + ((sin(s) - 2))^2), 0, 1);
q_y1 = sin(x);
q_x1 = cos(x);

fprintf('Closest point q to the point p using fminbnd: (%f, %f) \n',q_x1,q_y1);

% Part (e)

[q_x2, q_y2] = NR_min(@(s)cos(s), @(s)sin(s), 0.5, 4, 2);
fprintf('Closest point q to the point p using Newtons method: (%f, %f) \n',q_x2,q_y2);

% Part (f)

[q_x3, q_y3] = NR_min(@(s)(1 + 0.5 * cos(4 * s)) * cos(s), @(s)(1 + 0.5 * cos(4 * s)) * sin(s), 0, 1, 1);
[q_x4, q_y4] = NR_min(@(s)(1 + 0.5 * cos(4 * s)) * cos(s),@(s)(1 + 0.5 * cos(4 * s)) * sin(s), pi / 2, 1, 1);
[q_x5, q_y5] = NR_min(@(s)(1 + 0.5 * cos(4 * s)) * cos(s),@(s)(1 + 0.5 * cos(4 * s)) * sin(s), 5 * pi / 4, 1, 1);

s = linspace (0, 2 * pi);
x_s = (1 + 0.5 .* cos(4 * s)) .* cos(s);
y_s = (1 + 0.5 .* cos(4 * s)) .* sin(s);
hold on;
grid on;
xlabel('x(s)','fontsize',18);
ylabel('y(s)','fontsize',18);
plot(x_s, y_s,'k');
plot(q_x3, q_y3,'rd');
plot(q_x4, q_y4,'bd');
plot(q_x5, q_y5,'gd');
plot(1, 1,'mh');
legend('Parametric Interface', 'Point q for Initial Guess: 0', 'Point q for Initial Guess: \pi/2', 'Point q for Initial Guess: 5\pi/4','Point p (1,1)', 'location', 'southeastoutside','fontsize',18)
title('Q3(f): Figure with interface and solutions for q for different initial guesses','fontsize',18)
fprintf('Closest point q to the point p using Newtons method for Initial Guess s=0: (%f, %f) \n',q_x3,q_y3);
fprintf('Closest point q to the point p using Newtons method for Initial Guess s=pi/2: (%f, %f) \n',q_x4,q_y4);
fprintf('Closest point q to the point p using Newtons method for Initial Guess s=5*pi/4: (%f, %f) \n',q_x5,q_y5);