clear all
% close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
confs = [0.988483, 0.999995, 0.457370, 0.966003, 0.999038, 0.997176, 0.852632, 0.404166, 0.609248, 0.892175, 0.971515, 0.995673, 0.541367, 0.906055, 0.992315, 0.966639, 1.000000, 0.871618, 0.426503, 0.956077];
eps = [1, 5, 25, 100];

common = 'LLCI_eps%d/%s_stats.txt';
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(:,2)./confs(1,i);
    end
    Average(j,:) = mean(confidences,2);
end
figure
plot(1:128,Average, 'LineWidth', 1.5)
grid on
ylim([0 1])
xlabel('Noise standard deviation')
ylabel('Average True Class Confidence')
legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')


common = 'BI_eps%d/%s_stats.txt';
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(:,2)./confs(1,i);
    end
    Average(j,:) = mean(confidences,2);
end
figure
plot(1:128,Average, 'LineWidth', 1.5)
grid on
ylim([0 1])
xlabel('Noise standard deviation')
ylabel('Average True Class Confidence')
legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')


common = 'FGSM_eps%d/%s_stats.txt';
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(:,2)./confs(1,i);
    end
    Average(j,:) = mean(confidences,2);
end
figure
plot(1:128,Average, 'LineWidth', 1.5)
grid on
ylim([0 1])
xlabel('Noise standard deviation')
ylabel('Average True Class Confidence')
legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')