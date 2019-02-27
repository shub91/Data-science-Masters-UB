% Dr. Paul T. Bauman
%
% Goal: Function implements the Newton method for finding the roots
%       of a system of nonlinear equations.
%
% Input: func - function handle to nonlinear function
%        jacobian - function handle to the derivative of the nonlinear
%        function
%        x0 - vector, initial guess
%        tol - double, error tolerance
%        max_iters - integer, maximum number of iterations allowed.
%
% Output: x - vector, estimated root of nonlinear equation in func
%         error_est - double, value of the estimated error of our solution
%         num_iters - integer, number of iterations it took to converge

function [ x, error_est, num_iters ] = newton_sys( func, jacobian, x0, tol, max_iters )

% Set current solution iterate to the initial guess.
x = x0;

% Loop for the maximum number of iterations.
for i = 1:max_iters

    % Get the function value at the current solution iterate.
    f = func(x);
   
   % Get the error estimate.
   error_est(i) = norm( func(x) )/norm( func(x0) );

if( error_est(i) < tol )
        break
    end
 
    % Get the first derivative of the nonlinear function at 
    % the current solution iterate.
    J = jacobian(x);
   
    % Solve linear system 
    delta_x = J\f;
    
    % x_{i+1} = x_{i} - J^(-1){x_i}*f( x_{i} )
    x = x - delta_x;
    
end

num_iters = i;
