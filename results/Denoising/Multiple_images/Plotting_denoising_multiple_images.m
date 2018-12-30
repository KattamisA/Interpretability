% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
Confidence = zeros(101,6);


% load 'panda/Confidences.txt'
% Confidence(:,1) = Confidences(:,1);
% load 'peacock/Confidences.txt'
% Confidence(:,2) = Confidences(:,1);
% load 'F16_GT/Confidences.txt'
% Confidence(:,3) = Confidences(:,1);
% load 'monkey/Confidences.txt'
% Confidence(:,4) = Confidences(:,1);
% load 'zebra_GT/Confidences.txt'
% Confidence(:,5) = Confidences(:,1);
% load 'goldfish/Confidences.txt'
% Confidence(:,6) = Confidences(:,1);

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


Average = mean(Confidence,2);
Std = std(Confidence,0,2);
hold on
plot(t,Average,'b','LineWidth',1.5)
plot(t',[Average+Std, Average-Std],'--b','LineWidth',0.2,'HandleVisibility','off')
legend('Average denoising performance')
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')

% figure
% plot(t,Confidence)
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% grid on
% ylim([0,1.4])
