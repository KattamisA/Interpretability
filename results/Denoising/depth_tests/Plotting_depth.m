clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

common = 'depth%d/%s_PSNR.txt';
Value = zeros(51,20);
Average = zeros(51,7);
standard_deviation = zeros(51,7);

for j=1:7
    for i=1:size(img_names,2)
        path = sprintf(common,j,img_names(i));
        s = load(path);
        Value(:,i) = smooth(s(1:51,1),3);
    end
    Average(:,j) = mean(Value,2);
    standard_deviation(:,j) = std(Value,0,2);
end

figure
plot(0:100:5000, Average(:, 1:6), 'linewidth', 1.3)
xlabel('DIP iterations')
ylabel('PSNR')
grid on
legend('1-layer encoder','2-layer encoder','3-layer encoder','4-layer encoder','5-layer encoder - Baseline','6-layer encoder')%,'7-layer encoder')
ylim([10 32])

% figure
% plot(0:100:5000, standard_deviation, 'linewidth', 1)
% xlabel('DIP iterations')
% ylabel('PSNR standard deviation')
% grid on
% legend('Depth = 1','Depth = 2','Depth = 3', 'Depth = 4','Depth = 5 (Baseline)','Depth = 6')
