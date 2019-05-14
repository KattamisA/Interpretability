clear all
close all

%% Load and plot confidences
t=0:100:10000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

Confidence = zeros(101,size(img_names,2));
Confidence = zeros(101,size(img_names,2));
hold on

%% Calculating normalised
% common = 'Adam/%s_Normalised.txt';
common = 'Adam/lr%d/%s_Normalised.txt';
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,j,img_names(i));
        s = load(path);
        Confidence(:,i) = smooth(s(:,1),3);
        [Peak_amplitudes(j,i),Peak_iterations(j,i)]  = max(s(1:51,1));
    end
    Average(:,j) = mean(Confidence,2);
    Std(:,j) = std(Confidence,0,2);
%     Mean_iteration(j) = round(mean(Peak_iterations(j,:)));
    [M, Mean_iteration(j)] = max(Average(:,j));
    Mean_peak_amplitudes(j,:) = Confidence(Mean_iteration(j),:);    
    peak_mean_amps(j) = mean(Mean_peak_amplitudes(j,:));
    peak_std_amps(j) = std(Mean_peak_amplitudes(j,:));
    
    hold on
    plot([max(Mean_peak_amplitudes(j,:)),min(Mean_peak_amplitudes(j,:))],[j,j],'k','LineWidth',1,'HandleVisibility','off')
    plot([peak_mean_amps(j)-1*peak_std_amps(j),peak_mean_amps(j) + 1*peak_std_amps(j)],[j,j],'LineWidth',10)
    plot(Mean_peak_amplitudes(j,:),j,'xk','MarkerSize',8,'HandleVisibility','off')
end
    xlim([0,2])
    ylim([0,j+1])
    set(gca,'ytick',[])
    xlabel('Average Class Confidence')
    grid on
    legend('LR=0.001','LR=0.01','LR=0.1','LR=1')
%% Plotting

figure
hold on
plot(0:100:5000,Average(1:51,:),'LineWidth',1.3)
% plot(0:100:5000,[Average(1:51,:)+Std(1:51,:), Average(1:51,:)-Std(1:51,:)],'--r','LineWidth',0.2,'HandleVisibility','off')
xlabel('DIP iterations')
ylabel('True Class Confidence')
legend('LR=0.001','LR=0.01','LR=0.1','LR=1')
grid on
% plot(t,Std,'LineWidth',1)

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