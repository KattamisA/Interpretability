% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

Confidence = zeros(101,size(img_names,2));
Normalise = zeros(101,size(img_names,2));
hold on

Peak_iterations = zeros(1,size(img_names,2));
Peak_amplitudes = zeros(1,size(img_names,2));
Mean_peak_amplitudes = zeros(1,size(img_names,2));

%% Calculating normalised
common = '%s/%s_resnet50_Normalised.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i),img_names(i));
    s = load(path);
    Normalise(:,i) = s(:,1);
    [Peak_amplitudes(1,i),Peak_iterations(1,i)]  = max(s(1:101,1));
end
Average = mean(Normalise,2);
Std = std(Normalise,0,1);
Mean_iteration = round(mean(Peak_iterations(1,:)));
% [M,Mean_iteration] = max(Average);
Mean_peak_amplitudes(1,:) = Normalise(Mean_iteration,:);    
peak_mean_amps = mean(Mean_peak_amplitudes(1,:));
peak_std_amps = std(Mean_peak_amplitudes(1,:));

% hist(Peak_iterations(1,:).*100,250:500:9750)
% xlabel('DIP iterations')
% ylabel('Number of confidence peaks')
% grid on

% hold on
% plot([max(Mean_peak_amplitudes),min(Mean_peak_amplitudes)],[j,j],'k','LineWidth',1,'HandleVisibility','off')
% plot([peak_mean_amps-1*peak_std_amps,peak_mean_amps + 1*peak_std_amps],[j,j],'LineWidth',15)
% plot(Mean_peak_amplitudes(1,:),j,'xk','MarkerSize',7,'HandleVisibility','off')
% grid on
% ylim([0.5,2.5])
% xlim([0,2])
% set(gca,'ytick',[])
% xlabel('Average Class Confidence')

% figure
% plot(t,Normalise(:,19))

% Calculating Unnormalized
% common = 'EntropySGD_LR10/%s/Confidences.txt';
% for i=1:size(img_names,2)
%     path = sprintf(common,img_names(i));
%     q = load(path);
%     Confidence(:,i) = q(:,1);
%     [Peaks(1,i),Peaks(2,i)]  = max(q(1:51,1));
% end

%% Plotting
Average = mean(Normalise,2);
Std = std(Normalise,0,2);

% % plot(t,mean(Confidence,2))
hold on
plot(t,Average,'r','LineWidth',1.5)
plot(t,[Average+Std, Average-Std],'--r','LineWidth',0.2,'HandleVisibility','off')
% legend('Adam optimizer - average')
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')
% xlim([0 5000])
% plot(t,Std,'LineWidth',1)
grid on
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