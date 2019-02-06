% clear all
% close all

% %% Load and plot confidences
% t=0:100:10000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish'];%,'whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

Confidence = zeros(101,size(img_names,2));
Grads = zeros(101,size(img_names,2));
hold on

Peak_iterations = zeros(1,size(img_names,2));
Peak_amplitudes = zeros(1,size(img_names,2));
Mean_peak_amplitudes = zeros(1,size(img_names,2));

%% Calculating normalised
common = 'Gradients/Adam/%s_stats.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Grads(:,i) = abs(s(:,4));
end
Average = mean(Grads,2);

plot(100:100:size(Grads,1)*100, abs(Grads), 'LineWidth',2)
grid on
xlabel('DIP iterations')
ylabel("Absolute mean gradient")
figure
hold on
plot(100:100:size(Grads,1)*100, Average/Average(1), 'LineWidth',1)
grid on
xlabel('DIP iterations')
% ylabel("Averaged absolute mean gradient")
% figure
x = diff(Average/100);
plot(100:100:(size(Grads,1)-1)*100, x./x(1), 'LineWidth',1)
grid on
xlabel('DIP iterations')
% ylabel("First Derivative of mean gradient")
% figure
y = diff(x)/100;
plot(100:100:(size(Grads,1)-2)*100, y./y(1), 'LineWidth',1)
grid on
xlabel('DIP iterations')
% ylabel("Second Derivative of mean gradient")

ylabel('Normalized value')
legend('Actual value','First derivative','Second derivative')
%% Calculating Unnormalized
% common = 'EntropySGD_LR10/%s/Confidences.txt';
% for i=1:size(img_names,2)
%     path = sprintf(common,img_names(i));
%     q = load(path);
%     Confidence(:,i) = q(:,1);
%     [Peaks(1,i),Peaks(2,i)]  = max(q(1:51,1));
% end

%% Plotting
% Average = mean(Normalise,2);
% Std = std(Normalise,0,2);
% 
% % % plot(t,mean(Confidence,2))
% hold on
% plot(t,Average,'r','LineWidth',1.5)
% plot(t,[Average+Std, Average-Std],'--r','LineWidth',0.2,'HandleVisibility','off')
% % legend('Adam optimizer - average')
% xlabel('DIP iterations')
% ylabel('Averaged Class Confidence')
% % xlim([0 5000])
% % plot(t,Std,'LineWidth',1)
% grid on
% legend('Adam','EntropySGD')
%% Plotting all the lines 
% figure
% plot(t,Normalise)
% xlabel('DIP iterations')
% ylabel('Class Confidence')
%
% figure
% plot(t,Confidence)
% xlabel('DIP iterations')
% ylabel('Class Confidence')
% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')