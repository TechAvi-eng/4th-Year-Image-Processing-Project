clear
clc
close all


for i=[1:2:50]
load("0.mat")
im = rescale(im)+randn(size(im))*i/255;
save(num2str(i) )
end