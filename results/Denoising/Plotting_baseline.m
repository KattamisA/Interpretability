clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

Value = zeros(51,20);
Average = zeros(51,2);
standard_deviation = zeros(51,2);

common = 'Baseline/%s_PSNR.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Value(:,i) = s(1:51,1);
end
Average(:,1) = mean(Value,2);
standard_deviation(:,1) = std(Value,0,2);

common = 'EntropySGD/%s_PSNR.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Value(:,i) = s(1:51,1);
end
Average(:,2) = mean(Value,2);
standard_deviation(:,2) = std(Value,0,2);

figure
hold on
% plot(0:100:5000, Average, 'linewidth', 1)
plot(0:100:5000,Average(:,1),'b','LineWidth',1.5)
plot(0:100:5000,[Average(:,1)+standard_deviation(:,1), Average(:,1)-standard_deviation(:,1)],'--b','LineWidth',0.2,'HandleVisibility','off')
xlabel('DIP iterations')
ylabel('PSNR')
grid on
% legend('optimizer = Adam', 'optimizer = EntropySGD')

figure
plot(0:100:5000, standard_deviation, 'linewidth', 1)
xlabel('DIP iterations')
ylabel('PSNR standard deviation')
grid on
legend('optimizer = Adam', 'optimizer = EntropySGD')
