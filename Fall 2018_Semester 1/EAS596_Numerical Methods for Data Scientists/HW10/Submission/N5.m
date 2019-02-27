function [root,ic] = N5(f,Jacobian,intialGuess,maxIterations,tol)
ic = 0;
root = intialGuess;
error = 10000;
while (ic <= maxIterations)&&(error > tol)
    ic = ic +1;
     display(Jacobian(root));
     display(Jacobian(root));
    root = root - Jacobian(root)\f(root)';
    error = abs(norm(f(root))/norm(f(intialGuess)));
end