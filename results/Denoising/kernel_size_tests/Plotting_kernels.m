clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

common = 'kernel%d/%s_PSNR.txt';
Value = zeros(51,20);
Average = zeros(51,4);
standard_deviation = zeros(51,4);

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
plot(0:100:5000, Average(:,1:3), 'linewidth', 1.3)
ylim([10 32])
xlabel('DIP iterations')
ylabel('PSNR')
grid on
legend('Kernel size = 1-by-1', 'Kernel size = 3-by-3 (Baseline)', 'Kernel size = 5-by-5')

figure
plot(0:100:5000, standard_deviation, 'linewidth', 1)
xlabel('DIP iterations')
ylabel('PSNR standard deviation')
grid on
legend('Kernel size = 1x1', 'Kernel size = 3x3 (Baseline)', 'Kernel size = 5x5', 'Kernel size = 7x7')
