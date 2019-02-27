% Question 3

clc;
close;
clear;

% x = 2

x = 2;
tol = 0;

for i=1:1000
    func = (exp(-0.5*x)*(4-x))-2;
    der = -3*exp(-0.5*x) + 0.5*exp(-0.5*x)*x;
    div = func/der;
    xi1 = newton(x,div);
    err = abs(((exp(-0.5*xi1)*(4-xi1))-2) - ((exp(-0.5*x)*(4-x))-2));
    x = xi1;
    if(err < tol)
        break
    end
end

format short
fprintf("\nRoot with x = 2: %d", xi1);

% x = 6

x=6;

for i=1:1000
    func = exp(-0.5*x)*(4-x)-2;
    der = -3*exp(-0.5*x) + 0.5*exp(-0.5*x)*x;
    div = func/der;
        
    xi2 = newton(x,div);
    err = abs((exp(-0.5*xi2)*(4-xi2)-2) - (exp(-0.5*x)*(4-x)-2));
    
    if xi2 == Inf
        break
    else    
        x = xi2;
    end
    
    if(err < tol)
        break
    end
end
fprintf("\nRoot with x = 6: %d", xi2);

% x = 8

x=8;

for i=1:1000
    func = exp(-0.5*x)*(4-x)-2;
    der = -3*exp(-0.5*x) + 0.5*exp(-0.5*x)*x;
    div = func/der;
    xi3 = newton(x,div);

    err = abs((exp(-0.5*xi3)*(4-xi3)-2) - (exp(-0.5*x)*(4-x)-2));
       
    if xi3 == Inf
        break
    else    
        x = xi3;
    end

    if(err < tol)
       break
    end

end
format short
fprintf('\nRoot with x = 8: %d',xi3);

% The problem with Newton's method which is a derivative based method is
% that it doesnot gives a solution always. This is because in our Taylor
% series analysis in the derivation of the method we ignore the higher
% order terms under the assumption that our initial guess is close enough
% to the true solution. Thus, as we see here if the initial guess is far
% enough like x = 6,8 in this case the method diverges to infinity instead of
% converging. Our initial guess x = 2 was close enough to get us a solution
% here.
