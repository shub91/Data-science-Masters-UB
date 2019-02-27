function df = der(x, t)

% Function to find the derivative for non-uniform grid using central
% difference for the interior points and forward and backward difference
% for the end points
% Input arguments: x - vector of data points of variable whose derivative needs to be
% found, t - vector of data points of variable wrt to which derivate needs
% to be computed
% Output: Vector of data points of the derivative

len = length(x);
df = zeros(1, len);

df(1) = (x(2) - x(1)) / (t(2) - t(1));
df(len) = (x(len) - x(len - 1)) / (t(len) - t(len - 1));

for i = 2 : len - 1
    h1 = t(i + 1) - t(i);
    h0 = t(i) - t(i - 1);
    df(i) = (-1 * h1 * x(i - 1) / (h0 * (h0 + h1))) + ((h1 - h0) * x(i) / (h0 * h1)) + (h0 * x(i + 1) / (h1 * (h0 + h1)));
end

end