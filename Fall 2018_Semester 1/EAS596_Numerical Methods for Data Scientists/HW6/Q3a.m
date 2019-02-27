% Question 3(a)

% Linear Function

X = [0; 1; 2; 3; 4; 5; 6];
Y = [-0.02 ;1.1 ;1.98 ;3.05 ;3.95 ;5.1 ;6.02];
D = table(X, Y);
disp(D)
f = polyfit(X,Y,1);
v = polyval(f,X); 
D1 = table(X, v);
disp(D1)
figure
plot(X,Y,'o',X,v,'-') 
legend('Data','Linear Fit Model')
ye = Y - v;
disp(ye)
rssl = sum(ye.^2);
tssl = (length(Y)-1) * var(Y);
r2l = 1 - rssl/tssl
disp(r2l)

% The linear model f(x) = a0 + a1x is a better model. Although the R2 vaue
% for both is similar and is close to 1 which shows that our model has
% fitted the data quite well for both the cases. The relationships are
% true and overfitting is not happening. The solutions for both are very
% close since the coefficient for the x^2 term i.e. a2 is becoming 0 and is
% behaving like a linear model. Thus, it is better to keep less
% coefficients in our model and thus the linear model is better. 


