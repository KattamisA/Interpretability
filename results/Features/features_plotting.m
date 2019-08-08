clear all
close all

values = zeros(10,101);
labels = load('robust/labels.txt');
for i = 1:10
    name = sprintf('robust/%d_Confidences.txt',i-1);
    q = load(name);
    values(i,:) = q(:,labels(i)+1);
end

averaged(1,:) = mean(values,1);

plot(0:10:1000, averaged)
ylim([0 0.5])

values = zeros(10,101);
labels = load('non_robust/labels.txt');
for i = 1:10
    name = sprintf('robust/%d_Confidences.txt',i-1);
    q = load(name);
    values(i,:) = q(:,labels(i)+1);
end

averaged(1,:) = mean(values);

hold on
plot(0:10:1000, averaged)
ylim([0 0.2])