clear all
close all

values = zeros(10,5000);
for i = 1:10
    name = sprintf('robust/%d_stats.txt',i-1);
    q = load(name);
    values(i,1) = 