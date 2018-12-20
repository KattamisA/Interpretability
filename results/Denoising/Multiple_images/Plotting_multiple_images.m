clear all
close all

%% Load and plot confidences
t=0:100:10000;
Confidence = zeros(101,6);
hold on
load 'panda/Confidences.txt'
Confidence(:,1) = Confidences(:,1);
load 'peacock/Confidences.txt'
Confidence(:,2) = Confidences(:,1);
load 'F16_GT/Confidences.txt'
Confidence(:,3) = Confidences(:,1);
load 'monkey/Confidences.txt'
Confidence(:,4) = Confidences(:,1);
load 'zebra_GT/Confidences.txt'
Confidence(:,5) = Confidences(:,1);
load 'goldfish/Confidences.txt'
Confidence(:,6) = Confidences(:,1);

plot(t,mean(Confidence,2))

% plot(t,Confidence)
% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')