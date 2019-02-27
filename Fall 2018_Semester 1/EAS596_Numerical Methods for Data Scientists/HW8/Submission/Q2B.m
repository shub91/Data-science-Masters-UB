% Q-2(b)
function Q2B(f, p)
len = size(p);
len = len(2);
x = floor(sqrt(len));
y = len / x;
img = cell(1, len);
for i = 1:len
    subplot(x,y,i)
    img{i} = Q2(f, p(i));
    imshow(img{i});
    title("Compressed image for rank: " + p(i));
end
end