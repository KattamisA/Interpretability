clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

common = 'std%d/%s_PSNR.txt';
Value = zeros(51,20);
Average = zeros(51,7);
standard_deviation = zeros(51,7);

for j=1:7
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
legend('Iteration noise = 1/128','Iteration noise = 1/64','Iteration noise = 1/32','Iteration noise = 1/16','Iteration noise = 1/8','Iteration noise = 1/4','Iteration noise = 1/2')

figure
plot(0:100:5000, standard_deviation, 'linewidth', 1)
xlabel('DIP iterations')
ylabel('PSNR standard deviation')
grid on
legend('Iteration noise = 1/128','Iteration noise = 1/64','Iteration noise = 1/32','Iteration noise = 1/16','Iteration noise = 1/8','Iteration noise = 1/4','Iteration noise = 1/2')
