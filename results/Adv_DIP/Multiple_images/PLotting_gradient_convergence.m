% clear all
% close all

% %% Load and plot confidences
% t=0:100:10000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

Confidence = zeros(101,size(img_names,2));
Grads = zeros(101,size(img_names,2));
% hold on

Peak_iterations = zeros(1,size(img_names,2));
Peak_amplitudes = zeros(1,size(img_names,2));
Mean_peak_amplitudes = zeros(1,size(img_names,2));

%% Calculating normalised
common = 'Gradients/Adam/%s_stats.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Grads(:,i) = smooth(abs(s(:,4)),1);
end
Average = mean(Grads,2);

% loglog(1:100:(size(Grads,1)-1)*100+1, abs(Grads), 'LineWidth',2)
% grid on
% xlabel('DIP iterations')
% ylabel("Absolute mean gradient")
% figure
% hold on

Average = [Average(1:2); smooth(Average(3:end),11)];
% Average = wiener2(Average);
loglog(100:100:(size(Grads,1))*100, smooth(Average/Average(1),1), 'LineWidth',1.5)
grid on
xlim([100 10000])
xlabel('DIP iterations')
ylabel('Normalized value')







