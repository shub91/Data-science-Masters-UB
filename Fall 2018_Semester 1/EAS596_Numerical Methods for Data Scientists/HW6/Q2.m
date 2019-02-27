% Question 2(a.)
Qat = zeros(1,9);
Qbt = zeros(1,9);
Qct = zeros(1,9);
Qmt = zeros(1,9);
k = zeros(1,9);
for i = 2:10
    % display(i);
    A = 0;
    A = hilb(i);
    % display(A)
    k(i-1) = double (cond(A));
    % display(k)
    [Qm, Rm] = qr(A);
    display(Qm)
    Qmt(1,i-1) = norm(Qm'*Qm - eye(i)); % qr function
    [Qa, Ra] = Q1CGS(A);
    display(Qa)
    Qat(1,i-1) = norm(Qa'*Qa - eye(i)); % Classical GS
    [Qb, Rb] = Q1MGS(A);
    display(Qb)
    Qbt(1,i-1) = norm(Qb'*Qb - eye(i)); % Modified GS
    [Qc, Rc] = Q1HR(A);
    display(Qc)
    Qct(1,i-1) = norm((Qc'*Qc) - eye(i)); % Householder
end
display(k)
display(Qmt)
display(Qat)
display(Qbt)
display(Qct)
figure
plot(k, Qmt)
legend('QR Function norm vs Condition Number')
figure
plot(k, Qat)
legend('Classical GS norm vs Condition Number')
figure
plot(k, Qbt)
legend('Modified GS norm vs Condition Number')
figure
plot(k, Qct)
legend('Householder norm vs Condition Number')





% COMMENTS
% We know that as the order of the hilbert matrix increase the condition
% number will also increase. If we observe the norm of Q'Q-I for all the
% methods we find that as condition number increases to the order of 13th
% power of 10, my norm value for qr function remains minimum for all sizes
% of the hilbert matrix. After that comes the performance of Householder 
% algorithm which has the norm of the same order 10^-14, which is what we
% should expect since these algorithms always give good orthogonality even 
% for poorly conditioned matrices like these. The norm value for classical 
% and modified G-S is similar which shows poor orthogonality for 
% unconditioned matrices. Also the norm for all algorithms increase as the 
% condition number increses but for Householder algorithm we observe that
% after a certain point the norm decreses on increasing the condiction
% number.