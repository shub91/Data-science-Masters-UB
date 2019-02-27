% Question 2(a)
function a = Q2(f , p)
M = im2double(imread(f));
[x, y, z] = size(M);
if eq(z,1)
    M(:, :, 1) = M(:, :, 1);
    M(:, :, 2) = M(:, :, 1);
    M(:, :, 3) = M(:, :, 1);
end
[RU, RS, RV] = svd(M(:, :, 1));
[GU, GS, GV] = svd(M(:, :, 2));
[BU, BS, BV] = svd(M(:, :, 3));
RP = p;
GP = p;
BP = p;
if p > rank(M(:, :, 1))
    RP = rank(M(:, :, 1));
end
if p > rank(M(:, :, 2))
    GP = rank(M(:, :, 2));
end
if p > rank(M(:, :, 3))
    BP = rank(M(:, :, 3));
end
Rp = RU(1:x, 1:RP) * RS(1:RP, 1:RP) * RV(1:y, 1:RP)';
Gp = GU(1:x, 1:GP) * GS(1:GP, 1:GP) * GV(1:y, 1:GP)';
Bp = BU(1:x, 1:BP) * BS(1:BP, 1:BP) * BV(1:y, 1:BP)';
a = zeros(x, y, 3);
a(:, :, 1) = Rp;
a(:, :, 2) = Gp;
a(:, :, 3) = Bp;
end
