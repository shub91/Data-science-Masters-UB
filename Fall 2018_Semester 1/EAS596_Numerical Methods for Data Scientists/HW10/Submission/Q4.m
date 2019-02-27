% Question 4

clc;
clear;
close;

x = 5;
tol  = 0;
L = 100;
h = 97;

for i=1:1000
    func1 = (L- 2*x*sinh(h/(2*x)));
    func2 = ((h*cosh(h/(2*x)))/x - 2*sinh(h/(2*x)));
    div = func1/func2;
    XP = newton(x,div);
    err = abs((L- 2*XP*sinh(h/(2*XP))) - (L- 2*x*sinh(h/(2*x))));
    x = XP;
    if(err < tol)
        break
    end    
end

format short
fprintf('\na (as obtained from the Newton method): %d', XP);

x = linspace(-h/2,h/2,100);
y = XP*cosh(x/XP);

plot(x,y,'-.r');
title('Q4: Resulting Catenary Curve');
xlabel('x');
ylabel('y');
axis square;