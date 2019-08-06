clear all
close all

values = zeros(10,11);
labels = load('robust/labels.txt');
for i = 1:10
    name = sprintf('robust/%d_Confidences.txt',i-1);
    q = load(name);
    values(i,:) = q(:,labels(i)+1);
end

averaged(1,:) = mean(values);

plot(0:100:1000, averaged)