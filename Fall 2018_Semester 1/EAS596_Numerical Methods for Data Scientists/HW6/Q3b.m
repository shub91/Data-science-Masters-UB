% Question 3(b)

% Quadratic Polynomial

X = [0; 1; 2; 3; 4; 5; 6];
Y = [-0.02 ;1.1 ;1.98 ;3.05 ;3.95 ;5.1 ;6.02];
D = table(X, Y);
disp(D)
f = polyfit(X,Y,2);
v = polyval(f,X); 
D1 = table(X, v);
disp(D1)
figure
plot(X,Y,'o',X,v,'-') 
legend('Data','Quadratic Fit')
ye = Y - v;
disp(ye)
rss = sum(ye.^2);
tss = (length(Y)-1) * var(Y);
r2 = 1 - rss/tss
disp(r2)