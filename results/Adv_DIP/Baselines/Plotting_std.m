clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

Value = zeros(51,20);
Average = zeros(51,3);
standard_deviation = zeros(51,3);

common = 'Baseline/%s_Normalised.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Value(:,i) = smooth(s(:,1),3);
end
Average(:,1) = mean(Value,2);
standard_deviation(:,1) = std(Value,0,2);

common = 'EntropySGD_lr1/%s_Normalised.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Value(:,i) = smooth(s(:,1),3);
end
Average(:,2) = mean(Value,2);
standard_deviation(:,2) = std(Value,0,2);

common = 'EntropySGD_lr10/%s_Normalised.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Value(:,i) = smooth(s(:,1),3);
end
Average(:,3) = mean(Value,2);
standard_deviation(:,3) = std(Value,0,2);

figure
plot(0:100:5000, Average, 'linewidth', 1.3)
xlabel('DIP iterations')
ylabel('True Class Confidence')
xlim([0 5000])
ylim([0 1.2])
grid on
legend('Adam optimizer', 'EntropySGD optimizer with LR=1', 'EntropySGD optimizer with LR=10')

% figure
% plot(0:100:10000, standard_deviation, 'linewidth', 1)
% xlabel('DIP iterations')
% ylabel('PSNR standard deviation')
% xlim([0 5000]);
% grid on
% legend('Iteration noise = 1/128','Iteration noise = 1/64','Iteration noise = 1/32','Iteration noise = 1/16','Iteration noise = 1/8','Iteration noise = 1/4','Iteration noise = 1/2')
