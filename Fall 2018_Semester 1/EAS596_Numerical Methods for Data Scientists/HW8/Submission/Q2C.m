%Q-2(c)
close all; clear; clc;

sr = [1,2,7,24,70,199];
figure
Q2B("square.png", sr);

ubr = [1,10,25,68,145,401];
figure
Q2B("UB.png", ubr);

fr = [1,19,45,300,450,1213];
figure
Q2B("futurama.png", fr);

% The minimum rank which gives a good quality for square.png according to
% this computation is 2, for UB.png, it seems the image corresponding to
% rank 145 gives a good quality image, for futurama.png, the minimum rank
% is 300.

% Clearly the images which have more edges will have higher ranks for a
% good image clarity since that correspond to sudden changes in the
% graysacle whereas the square image has very low rank for a good image
% quality since it has a constant color throughout the image. Certainly,
% the rank would increses as the pixel size will get small as we can see in
% the UB and futurama example.