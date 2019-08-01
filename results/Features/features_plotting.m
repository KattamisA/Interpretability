clear all
close all

values = zeros(10,11);
for i = 1:10
    name = sprintf('non_robust/%d_Confidences.txt',i-1);
    q = load(name);
    values(i,:) = q(:,1);
end

averaged(1,:) = mean(values);
% 
% values = zeros(10,51);
% for i = 1:10
%     name = sprintf('non_robust/%d_Normalised.txt',i-1);
%     q = load(name);
%     values(i,:) = q(:,1);
% end
% 
% averaged(2,:) = mean(values);
% 
% plot(0:100:5000,averaged)

plot(0:100:1000, averaged)