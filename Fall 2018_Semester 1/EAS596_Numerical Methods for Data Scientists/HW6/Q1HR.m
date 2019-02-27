% Question 1(3.)

function [Q,R] = Q1HR(A)
[m, n] = size(A);
Q = eye(m);
R = A;

for k = 1:n
    lenx = norm(R(k:end,k));
    x = R(k,k) - sign(R(k,k))*lenx;
    y = R(k:end,k)/x;
    y(1) = 1;
    t = -sign(R(k,k))*x/lenx;
    R(k:end,:) = R(k:end,:)-(t*y)*(y'*R(k:end,:));
    Q(:,k:end) = Q(:,k:end)-(Q(:,k:end)*y)*(t*y)';
    
end

end