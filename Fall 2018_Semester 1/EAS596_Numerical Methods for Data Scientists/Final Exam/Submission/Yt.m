
function D = Yt(t,y)
% Function to define the function given in the question
% Input: t,y - the variables
% Output: The derivative of y wrt t as given in the function
    if t == 0
        D = 1;
    else
        D = y*(-2*t + 1/t);
    end
end