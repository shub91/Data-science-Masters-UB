% Question 3
% Part a
clear; close all; clc;
SV = load('senate_votes.txt');
SP = load('senate_parties.txt');
S = fopen('senate_names.txt');
SN = textscan(S,'%s %s');

%Part b
sigma = svd(SV);
figure(1);
plot(sigma);
title('Singular Values');
ylabel('Singular Values');

%Part c
[U,S,V] = svd(SV);
rank(SV);
D = find(SP == 100); % finding party affiliation
R = find(SP == 200);
n = repelem(7,size(U,1));
n(D) = 1;
n(R) = 5;
% display((n));
figure(2)
scatter(U(:,1),U(:,2),[],n);
title('(U1, U2) by Party Affiliation: Green - Republic, Blue - Democrat, Yellow - Independent');

figure(3)
scatter(U(:,1),U(:,2),[],n);
text(U(:,1),U(:,2), SN{1}');
title('Names: (U1, U2) by Party Affiliation: Green - Republic, Blue - Democrat, Yellow - Independent');

% Part d
figure(4)
scatter(U(:,1),U(:,3),[],n);
title('(U1, U3) by Party Affiliation: Green - Republic, Blue - Democrat, Yellow - Independent');

figure(5)
scatter(U(:,2),U(:,4),[],n);
title('(U2, U4) by Party Affiliation: Green - Republic, Blue - Democrat, Yellow - Independent');

figure(6)
scatter(U(:,2),U(:,3),[],n);
title('(U2, U3) by Party Affiliation: Green - Republic, Blue - Democrat, Yellow - Independent');

figure(7)
scatter(U(:,6),U(:,8),[],n);
title('(U6, U8) by Party Affiliation: Green - Republic, Blue - Democrat, Yellow - Independent');

% Part e

r2 = S(1,1)*U(:,1)*V(:,1)' + S(2,2)*U(:,2)*V(:,2)';

CV = repelem(0,size(U,1)); % NO of correct votes
for i = 1:size(r2,1)
    for j = 1:size(r2,2)
        if(r2(i,j) > 0)
            r2(i,j) = 1;
            if(SV(i,j) == 1)
               CV(i) = CV(i)+1; 
            end
        else
            r2(i,j) = -1;
            if(SV(i,j) == -1)
               CV(i) = CV(i)+1;
            end
        end
    end
end

figure(8)
plot(CV, U(:,1)');
text(CV, U(:,1)', SN{1}');
title('No. of matches vs the U1 vector');

figure(9)
scatter(CV, U(:,1)',[],n);
text(CV, U(:,1)', SN{1}');
title('No. of matches vs the U1 vector');
fprintf('No. of Correct Votes: \n')
disp(sum(CV));

% Part f
HV = load('house_votes.txt');
HP = load('house_parties.txt');
H = fopen('house_names.txt');
HN = textscan(H,'%s %s');

