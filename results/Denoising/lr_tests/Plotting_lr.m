clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

common = 'lr%d/%s_PSNR.txt';
Value = zeros(51,20);
Average = zeros(51,7);
standard_deviation = zeros(51,7);

for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,j,img_names(i));
        s = load(path);
        Value(:,i) = s(1:51,1);
    end
    Average(:,j) = mean(Value,2);
    standard_deviation(:,j) = std(Value,0,2);
end

figure
plot(0:100:5000, Average, 'linewidth', 1)
xlabel('DIP iterations')
ylabel('PSNR')
grid on
legend('Learning rate = 0.001','Learning rate = 0.01 (Baseline)','Learning rate = 0.1', 'Learning rate = 1')

figure
plot(0:100:5000, standard_deviation, 'linewidth', 1)
xlabel('DIP iterations')
ylabel('PSNR standard deviation')
grid on
legend('Learning rate = 0.001','Learning rate = 0.01 (Baseline)','Learning rate = 0.1', 'Learning rate = 1')
