clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
confs = [0.988483, 0.999995, 0.457370, 0.966003, 0.999038, 0.997176, 0.852632, 0.404166, 0.609248, 0.892175, 0.971515, 0.995673, 0.541367, 0.906055, 0.992315, 0.966639, 1.000000, 0.871618, 0.426503, 0.956077];

% hold on
% common = 'Epsilon_test/%s_fgsm.txt';
% % Average = zeros(52,20);
% for i=1:size(img_names,2)
%     path = sprintf(common,img_names(i));
%     s = load(path);
%     Average(2:end,i) = smooth(s(1:51,2),1);
% end
% Averaged = mean(Average,2);
% Averaged(1) = sum(confs)/20;
% plot([0,1:2:101], Averaged, 'linewidth', 1)
% 
% common = 'Epsilon_test/%s_bi.txt';
% for i=1:size(img_names,2)
%     path = sprintf(common,img_names(i));
%     s = load(path);
%     Average(2:end,i) = smooth(s(1:51,2),1);
% end
% Averaged = mean(Average,2);
% Averaged(1) = sum(confs)/20;
% plot([0,1:2:101], Averaged, 'linewidth', 1)
% 
% common = 'Epsilon_test/%s_llci.txt';
% for i=1:size(img_names,2)
%     path = sprintf(common,img_names(i));
%     s = load(path);
%     Average(2:end,i) = smooth(s(1:51,2),1);
% end
% Averaged = mean(Average,2);
% Averaged(1) = sum(confs)/20;
% plot([0,1:2:101], Averaged, 'linewidth', 1)
% 
% grid on
% legend('FGSM', 'BI', 'LLCI')

q = 1:100;
qs = round(max(1.25*q,q+4));

figure
common = 'jsma/%s.txt';
Average = zeros(126,20);
Least = zeros(126,20);
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Average(2:end,i) = s(:,2);
    Least(2:end,i) = s(:,3);
end
Averaged = mean(Average,2);
Least = mean(Least,2);
Averaged(1) = sum(confs)/20;
plot(0:100, [Averaged([1,qs]), Least([1,qs])], 'linewidth', 1)
grid on
xlabel('Adversarial strength, \epsilon')
ylabel('Classification confidence')
legend('Original class','Least Likely Class')
