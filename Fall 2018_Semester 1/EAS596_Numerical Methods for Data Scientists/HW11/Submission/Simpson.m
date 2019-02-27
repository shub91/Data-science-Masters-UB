function R = Simpson(a, b, steps, func)
  h = (b - a) / steps; 
  x = a : h : b;
  R = h / 3 * (func(x(1)) + 2 * sum(func(x( 3 : 2 : end - 2))) + 4 * sum(func(x( 2 : 2 : end))) + func(x(end)));
end