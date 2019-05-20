clear all
close all

%% Load and plot confidences
t=0:100:5000;
% img_names = ["panda", "F16_GT", "monkey",'zebra_GT'];
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
Confidence = zeros(51,size(img_names,2));
common = 'Adam/%s/%s_Normalised.txt';

tests = [1, 2, 3, 4, 5 ,6, 7];
Averaged= zeros(51,size(tests,2));
Std = zeros(51,size(tests,2));
Peak_iterations = zeros(size(tests,2),size(img_names,2));
Peak_amplitudes = zeros(size(tests,2),size(img_names,2));
Mean_peak_amplitudes = zeros(size(tests,2),size(img_names,2));
Mean_iteration = zeros(size(tests,2));
%% Filling in the matrices
for j=1:size(tests,2)
    for i=1:size(img_names,2)
        test = sprintf('depth%d',tests(j));
        path = sprintf(common,test,img_names(i));
        s = load(path);
        Confidence(:,i) = smooth(s(1:51,1),3);
        [Peak_amplitudes(j,i),Peak_iterations(j,i)]  = max(s(1:51,1));
    end
    Averaged(:,j) = smooth(mean(Confidence,2),1);
    Std(:,j) = std(Confidence,0,2);
%     Mean_iteration(j) = round(mean(Peak_iterations(j,:)));
    [M,Mean_iteration(j)] = max(Averaged(:,j));
    Mean_peak_amplitudes(j,:) = Confidence(Mean_iteration(j),:);    
    peak_mean_amps(j) = mean(Mean_peak_amplitudes(j,:));
    peak_std_amps(j) = std(Mean_peak_amplitudes(j,:));
    
%     hold on
%     plot([max(Mean_peak_amplitudes(j,:)),min(Mean_peak_amplitudes(j,:))],[j,j],'k','LineWidth',1,'HandleVisibility','off')
%     plot([peak_mean_amps(j)-1*peak_std_amps(j),peak_mean_amps(j) + 1*peak_std_amps(j)],[j,j],'LineWidth',10)
%     plot(Mean_peak_amplitudes(j,:),j,'xk','MarkerSize',8,'HandleVisibility','off')
% 
%     xlim([0,2])
% %     ylim([0,8])
%     set(gca,'ytick',[])
%     xlabel('Average Class Confidence')
%     hist(Peak_iterations(j,:),0:2:50)
end
% ylim([0,size(tests,2)+2])
% grid on
% legend('Depths = 1','Depths = 2', 'Depths = 3', 'Depths = 4','Depths = 5 (Baseline)','Depths = 6')

%% Plotting
% plot(t,Std,'LineWidth',1)
% legend('show')
figure
hold on
% Averaged = Averaged(:,2);
% Std = Std(:,2);
plot(t,Averaged,'LineWidth',1.3)
% plot(t,[Averaged+Std, Averaged-Std],'--b','LineWidth',0.2,'HandleVisibility','off')
% legend('Adam optimizer - average')
legend('1-layer encoder','2-layer encoder', '3-layer encoder', '4-layer encoder','5-layer encoder (Baseline)','6-layer encoder', '7-layer encoder')
xlabel('DIP iterations')
ylabel('True Class Confidence')
xlim([0 5000])
% % plot(t,Std,'LineWidth',1)
grid on
% legend('Constant Feature Maps - # parameters ~150k','Changing Feature Maps - # parameters ~150k', 'Constant Feature Maps - # parameters ~600k', 'Changing Feature Maps - # parameters ~600k','Constant Feature Maps - # parameters ~2200k(Baseline)','Changing Feature Maps - # parameters ~2200k')
