function R = Trapez(a, b, steps, func)
R = 0; 
h = (b - a) / steps; 
x = a + h : h : b - h; 
for i = 1 : steps - 1
    R = R + func(x(i));
end
R = h * (R + 0.5 * (func(a) + func(b)));
end