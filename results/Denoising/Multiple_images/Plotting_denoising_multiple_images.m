% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
Confidence = zeros(101,6);

Confidence = zeros(101,6);
load 'panda/Confidences.txt'
Confidence(:,1) = Confidences(:,1)./0.98848;
load 'peacock/Confidences.txt'
Confidence(:,2) = Confidences(:,1)./1;
load 'F16_GT/Confidences.txt'
Confidence(:,3) = Confidences(:,1)./0.45737;
load 'monkey/Confidences.txt'
Confidence(:,4) = Confidences(:,1)./0.966;
load 'zebra_GT/Confidences.txt'
Confidence(:,5) = Confidences(:,1)./0.99904;
load 'goldfish/Confidences.txt'
Confidence(:,6) = Confidences(:,1)./0.99718;

Confidence = zeros(101,6);
load 'EntropySGD/panda/Confidences.txt'
Confidence(:,1) = Confidences(:,1)./0.98848;
load 'EntropySGD/peacock/Confidences.txt'
Confidence(:,2) = Confidences(:,1)./1;
load 'EntropySGD/F16_GT/Confidences.txt'
Confidence(:,3) = Confidences(:,1)./0.45737;
load 'EntropySGD/monkey/Confidences.txt'
Confidence(:,4) = Confidences(:,1)./0.966;
load 'EntropySGD/zebra_GT/Confidences.txt'
Confidence(:,5) = Confidences(:,1)./0.99904;
load 'EntropySGD/goldfish/Confidences.txt'
Confidence(:,6) = Confidences(:,1)./0.99718;


Average = mean(Confidence,2);
Std = std(Confidence,0,2);
hold on
plot(0:100:5000,Average(1:51,:),'r','LineWidth',1.5)
plot(0:100:5000',[Average(1:51,:)+Std(1:51,:), Average(1:51,:)-Std(1:51,:)],'--r','LineWidth',0.2,'HandleVisibility','off')
legend('Average denoising performance')
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')

% figure
% plot(t,Confidence)
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% grid on
% ylim([0,1.4])
