% Question 1

clc;
close;
clear;

% Part a
Y0 = 0;
TSPAN = [0,1];
[TOUT,YOUT] = ode45(@Yt,TSPAN,Y0);
fprintf('The solution obtained is: %f at time step number %f \n',YOUT(numel(YOUT)),numel(TOUT));

% Part d

n = 2;
err = 1;
acc = 0.01;
Yt_2 = @(t)(t * exp(-t*t) * (-2*t + 1/t));
init = YOUT(numel(YOUT));

while (err > acc)
    n = n+2;
    fin = gauss_quad_2pt(Yt_2,0,1,n);
    err = (fin - init)/init; % within 1% of the exact solution
end
fprintf('The solution obtained using 2 point guass quadrature rule is: %f for number of guass intervals =  %f \n',fin,n);


