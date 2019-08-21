clear all
close all

values = zeros(10,101);
for i = 1:10
    name = sprintf('robust/%d_stats.txt',i-1);
    q = load(name);
    values(i,:) = q(:,2);
end

averaged(1,:) = mean(values,1);

plot(0:10:1000, averaged, 'Linewidth', 1.25)
% ylim([0 0.5])

values = zeros(10,101);
for i = 1:10
    name = sprintf('non_robust/%d_stats.txt',i-1);
    q = load(name);
    values(i,:) = q(:,2);
end

averaged(1,:) = mean(values);

hold on
plot(0:10:1000, averaged, 'Linewidth', 1.25)
% ylim([0 0.2])
xlim([0 200])
grid on
legend('Robust images','Non-robust images')