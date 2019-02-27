function y = Lag_fn(x, xp, yp)

% Defining the lagrange polynomial for xp and yp vector
len = length(xp);
y = 0;
for i = 1 : len
    y = y + yp(i) * Lagr_fn(i, x, xp);
end

end