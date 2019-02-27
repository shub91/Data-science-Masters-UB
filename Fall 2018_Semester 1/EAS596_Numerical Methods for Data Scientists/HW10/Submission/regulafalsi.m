%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%regular falsi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = regf(a,b,f)
x = b - ((f(b) * (a - b)) / (f(a) - f(b)));
end