clc;
close;
clear;
% QUESTION 2

a = 0.5;
b = 1;
tol = 1.0 * 10^(-6);

% Q 2a
n=1000;

BE = [];
BT = [];

f1=fun(a);
f2=fun(b);
    
for i=1:n
     
        mid = bisection(a,b);
        funcmid = fun(mid);
        
            if(abs(funcmid) < tol)
                break
            end
            
            BE = [BE, abs(funcmid)];
            BT = [BT, i];
        
            if((f1*funcmid)<0)
                b=mid;
                f2 = fun(b);
            else
                a=mid;
                f1 = fun(a);
            end
            
            mid = bisection(a,b);
            funcmid = fun(mid);
            
   
end
fprintf('Root of the non-linear equation using Bisection method: %d',mid);

%Q2b

a = 0.5;
b = 1;

RFerr = [];
RFiter = [];

n = 1000;
for i = 1:n
    
    cal = regulafalsi(a,b,@fun);
    funcmid = fun(cal);
    
    RFerr = [RFerr, abs(funcmid)];
    RFiter = [RFiter, i];
    
    if abs(funcmid) < tol
        break;
    end
    
    if funcmid * fun(a) > 0
        a = cal;
    else
        b = cal;
    end
end

fprintf("\nRoot of the non-linear equation using Regula Falsi method: %d", cal);

%Q2c

b=1;
NEerr = [];
NEiter = [];
n =1000;

for i=1:n
    f1 = fun(b);
    f2 = cos(b)-(3*b^2);
    divfunc = f1/f2;
    xiplus = newton(b, divfunc);
    
    NEerr  = [NEerr, abs(fun(xiplus) - fun(b))];
    NEiter = [NEiter, i];
    
    error = abs(fun(xiplus) - fun(b));
    b = xiplus;
    
    if(error < tol)
        break
    end
end

fprintf('\nRoot of the non-linear equation using Newton method: %d',b);

figure;
semilogy(BE,'--g');
hold on;
semilogy(RFerr,'-.b');
hold on;
semilogy(NEerr,':r');
title('Ques 2: Error vs. No. of Iterations');
xlabel('Number  of Iterations');
ylabel('Error');
legend('Bisection method', 'Regular Falsi method', 'Newton method',"Location","eastoutside");
grid on;
