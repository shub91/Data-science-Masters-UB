%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ans 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
close;
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Applied ODE45 Runge Kutta

[t,y] = ode45(@(t,y) func(t,y),[0,1],0);

%Please note that this plot has not been asked in the question,
%it is just to bring more clarity
plot(t,y,'r-');
xlabel('Time Steps Taken');
ylabel('Output Value using function');
title('Application of the Runge Kutta on the given function')

%Thus, upon looking at the table of 't', we can say that
%no. of timestepsn9 taken is '57'.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part d
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 2;
error = 1;
tol = 0.01;

func2 = @(t)(t*exp(-t*t)*(-2*t+1/t));
fi = gauss_quad(func2,0,1,n);

while (error > tol)
    
    %incrementing the value of n
    n = n+2;
    %implementing the gaussian quadrature again and again, till a tolerance
    %level
    ff = gauss_quad(func2,0,1,n);
    %calculating the error value
    error = (ff - fi)/ff;
    %setting the new function value as the initial value, to calculate the
    %error
    fi = ff;
    
end

val_int = n
fin_val  = ff


