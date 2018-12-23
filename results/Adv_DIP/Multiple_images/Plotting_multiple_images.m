clear all
close all

%% Load and plot confidences
t=0:100:10000;
Confidence = zeros(101,18);
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
Confidence(:,7) = Confidences(:,1);
load 'whale/Confidences.txt'
Confidence(:,8) = Confidences(:,1);
load 'dolphin/Confidences.txt'
Confidence(:,9) = Confidences(:,1);
load 'spider/Confidences.txt'
Confidence(:,10) = Confidences(:,1);
load 'labrador/Confidences.txt'
Confidence(:,11) = Confidences(:,1);
load 'snake/Confidences.txt'
Confidence(:,12) = Confidences(:,1);
load 'flamingo_animal/Confidences.txt'
Confidence(:,13) = Confidences(:,1);
load 'canoe/Confidences.txt'
Confidence(:,14) = Confidences(:,1);
load 'car_wheel/Confidences.txt'
Confidence(:,15) = Confidences(:,1);
load 'fountain/Confidences.txt'
Confidence(:,16) = Confidences(:,1);
load 'football_helmet/Confidences.txt'
Confidence(:,17) = Confidences(:,1);
load 'hourglass/Confidences.txt'
Confidence(:,18) = Confidences(:,1);
load 'refrigirator/Confidences.txt'
Confidence(:,6) = Confidences(:,1);

plot(t,mean(Confidence,2))

% plot(t,Confidence)
% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')