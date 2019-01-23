clear all
% close all

%% Load and plot confidences
t=0:100:10000;
% img_names = ["panda", "F16_GT", "monkey",'zebra_GT'];
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador'];
Confidence = zeros(101,size(img_names,2));
common = 'EntropySGD/%s_%d_256_Normalised.txt';

std1 = [1, 2, 4, 8];
Averaged= zeros(101,size(std1,2));
Std = zeros(101,size(std1,2));
Peak_iterations = zeros(size(std1,2),size(img_names,2));
Peak_amplitudes = zeros(size(std1,2),size(img_names,2));
Mean_peak_amplitudes = zeros(size(std1,2),size(img_names,2));
Mean_iteration = zeros(size(std1,2),1);
%% Filling in the matrices
for j=1:size(std1,2)
    for i=1:size(img_names,2)
        path = sprintf(common,img_names(i),std1(j));
        s = load(path);
        Confidence(:,i) = s(:,1);
        [Peak_amplitudes(j,i),Peak_iterations(j,i)]  = max(s(1:51,1));
    end
    Averaged(:,j) = smooth(mean(Confidence,2));
    Std(:,j) = std(Confidence,0,2);
%     Mean_iteration(j) = round(mean(Peak_iterations(j,:)));
    [M,Mean_iteration(j)] = max(Averaged(:,j));
    Mean_peak_amplitudes(j,:) = Confidence(Mean_iteration(j),:);    
    peak_mean_amps(j) = mean(Mean_peak_amplitudes(j,:));
    peak_std_amps(j) = std(Mean_peak_amplitudes(j,:));
    
    hold on
    plot([peak_mean_amps(j)-1*peak_std_amps(j),peak_mean_amps(j) + 1*peak_std_amps(j)],[j,j],'LineWidth',20)
    plot(Mean_peak_amplitudes(j,:),j,'xk','MarkerSize',8)
end
ylim([0,size(std1,2)+1])
grid on



%% Plotting
% plot(t,Std,'LineWidth',1)
% legend('show')
% figure
% hold on
% plot(0:100:5000,Averaged(1:51,3),'b','LineWidth',1.5)
% plot(0:100:5000,[Averaged(1:51,3)-Std(1:51,3), Averaged(1:51,3)+Std(1:51,3)],'--b','LineWidth',0.2,'HandleVisibility','off')
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% legend('\sigma_{noise} = 1/64','\sigma_{noise} = 1/32','\sigma_{noise} = 1/16','\sigma_{noise} = 1/8','\sigma_{noise} = 1/4','\sigma_{noise} = 1/2')
% legend('Std = 1/256','Std = 1/128','Std = 1/64','Std = 1/32');
grid on
