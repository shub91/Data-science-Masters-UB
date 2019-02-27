clc;
close;
clear;

% Question 1

a = 1;
v = 1;
N = 21;

% Considering Part 1(b)

A = full(gallery('tridiag',N,-(a*N + 2*v*N^2),(4*v*N^2),(a*N - 2*v*N^2)));
B = repelem(2,N);
B(1) = 0 ;
B(21) = 0;
Y = A\B';
x = linspace(0,1,N);
u = ((1/(a*(1-exp(a/v))))*exp(a*x/v) + (x/a) - (1/(a*(1-exp(a/v))))); 

figure;
plot(x,u);
hold on
plot(x,Y);
legend("Exact Solution" , "Numerical Approximation", "Location","eastoutside")
xlabel('x');
ylabel('u(x)');
title('a=v=1, Using Central Difference Approximation for both terms');



a1 = 500;
v = 1;

A=full(gallery('tridiag',N,-(a1*N + 2*v*N^2),(4*v*N^2),(a1*N - 2*v*N^2)));
B = repelem(2,N);
B(1) = 0 ;
B(21) = 0;
Y = A\B';  % Numerical Approximation
x = linspace(0,1,N);
u = ((1/(a1*(1-exp(a1/v))))*exp(a1*x/v) + (x/a1) - (1/(a1*(1-exp(a1/v)))));

figure;
plot(x,u);
hold on
plot(x,Y);
legend("Exact Solution" , "Numerical Approximation", "Location","eastoutside")
xlabel('x');
ylabel('u(x)');
title('a=500, v=1, Using Central Difference Approximation for both terms');

% Considering Part 1(c)

A=full(gallery('tridiag',N,(a*N - 2*v*N^2),(4*v*N^2-a*N),-(2*v*N^2)));
B = repelem(2,N);
B(1) = 0 ;
B(21) = 0;
Y = A\B';
x = linspace(0,1,N);
u = ((1/(a*(1-exp(a/v))))*exp(a*x/v) + (x/a) - (1/(a*(1-exp(a/v)))));

figure;
plot(x,u);
hold on
plot(x,Y);
legend("Exact Solution" , "Numerical Approximation", "Location","eastoutside")
xlabel('x');
ylabel('u(x)');
title('a=v=1, Using Central Difference and Backward difference');


A=full(gallery('tridiag',N,(a1*N - 2*v*N^2),(4*v*N^2-a1*N),-(2*v*N^2)));
B = repelem(2,N);
B(1) = 0 ;
B(21) = 0;
Y = A\B';
x = linspace(0,1,N);
u = ((1/(a1*(1-exp(a1/v))))*exp(a1*x/v) + (x/a1) - (1/(a1*(1-exp(a1/v)))));

figure;
plot(x,u);
hold on
plot(x,Y);
legend("Exact Solution" , "Numerical Approximation", "Location","eastoutside")
xlabel('x');
ylabel('u(x)');
title('a=500, v=1, Using Central Difference and Backward difference');