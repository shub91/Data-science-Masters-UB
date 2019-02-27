% Question 4

m=50;
n=12;
t=linspace(0,1,m);
b=(cos(4*t))';
A=zeros(m,n);
A(:,1)=1;
for i=2:n
    for j=1:m
        A(j,i)=t(j)^(i-1);
    end
end

% Normal equations
cn=inv(A'*A)*A'*(b);

% Modified G-S
[Qmgs, Rmgs]=Q1MGS(A);
cm = Rmgs\Qmgs.'*b;

% Householder Reflector function
[Qh, Rh]=Q1HR(A);
ch = Rh\Qh.'*b;

% qr function
[Qq Rq]=qr(A);
cq = Rq\Qq.'*b;

% A\b function
ca=A\b;

figure
plot(cn,'+')
hold on
plot(cm,'*')

hold on
plot(ch,'o')

hold on
plot(cq,'r*')

plot(ca,'go')
legend('Normal equation coefficients','Modified G-S coefficients','Householder Reflector coefficients','qr function coefficients','A\b function coefficients')

% COMMENTS

% We observe that for each model coefficients obtained are more or less
% very close in value as is clear from the graph. We are fitting a linear
% model and trying to simplify the solution using using differet methods.
% We observe very close agrrement in the solutions from different models
% which is what we should expect.