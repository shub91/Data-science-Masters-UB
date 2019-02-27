function lagr = Lagr_fn(i, x, xp)

% Defining Lagrange's polynomial terms
len = length(xp);
lagr = 1;

for j = 1 : len
    if i ~= j
        lagr = lagr .* (x - xp(j)) / (xp(i) - xp(j));
    end
end

end