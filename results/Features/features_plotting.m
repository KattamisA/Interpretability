clear all
close all

values = zeros(10,51);
for i = 1:10
    name = sprintf('robust/%d_stats.txt',i-1);
    q = load(name);
    values(i,:) = q(:,2);
end

averaged(1,:) = mean(values);

values = zeros(10,51);
for i = 1:10
    name = sprintf('non_robust/%d_stats.txt',i-1);
    q = load(name);
    values(i,:) = q(:,2);
end

averaged(2,:) = mean(values);

plot(0:100:5000,averaged)