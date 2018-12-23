clear all
close all

%% Load and plot confidences
t=0:100:10000;
Confidence = zeros(101,19);
hold on

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
% Confidence(:,7) = Confidences(:,1);
% load 'whale/Confidences.txt'
% Confidence(:,8) = Confidences(:,1);
% load 'dolphin/Confidences.txt'
% Confidence(:,9) = Confidences(:,1);
% load 'spider/Confidences.txt'
% Confidence(:,10) = Confidences(:,1);
% load 'labrador/Confidences.txt'
% Confidence(:,11) = Confidences(:,1);
% load 'snake/Confidences.txt'
% Confidence(:,12) = Confidences(:,1);
% load 'flamingo_animal/Confidences.txt'
% Confidence(:,13) = Confidences(:,1);
% load 'canoe/Confidences.txt'
% Confidence(:,14) = Confidences(:,1);
% load 'car_wheel/Confidences.txt'
% Confidence(:,15) = Confidences(:,1);
% load 'fountain/Confidences.txt'
% Confidence(:,16) = Confidences(:,1);
% load 'football_helmet/Confidences.txt'
% Confidence(:,17) = Confidences(:,1);
% load 'hourglass/Confidences.txt'
% Confidence(:,18) = Confidences(:,1);
% load 'refrigirator/Confidences.txt'
% Confidence(:,6) = Confidences(:,1);

load 'panda/Normalised.txt'
Confidence(:,1) = Normalised(:,1);
load 'peacock/Normalised.txt'
Confidence(:,2) = Normalised(:,1);
load 'F16_GT/Normalised.txt'
Confidence(:,3) = Normalised(:,1);
load 'monkey/Normalised.txt'
Confidence(:,4) = Normalised(:,1);
load 'zebra_GT/Normalised.txt'
Confidence(:,5) = Normalised(:,1);
load 'goldfish/Normalised.txt'
Confidence(:,7) = Normalised(:,1);
load 'whale/Normalised.txt'
Confidence(:,8) = Normalised(:,1);
load 'dolphin/Normalised.txt'
Confidence(:,9) = Normalised(:,1);
load 'spider/Normalised.txt'
Confidence(:,10) = Normalised(:,1);
load 'labrador/Normalised.txt'
Confidence(:,11) = Normalised(:,1);
load 'snake/Normalised.txt'
Confidence(:,12) = Normalised(:,1);
load 'flamingo_animal/Normalised.txt'
Confidence(:,13) = Normalised(:,1);
load 'canoe/Normalised.txt'
Confidence(:,14) = Normalised(:,1);
load 'car_wheel/Normalised.txt'
Confidence(:,15) = Normalised(:,1);
load 'fountain/Normalised.txt'
Confidence(:,16) = Normalised(:,1);
load 'football_helmet/Normalised.txt'
Confidence(:,17) = Normalised(:,1);
load 'hourglass/Normalised.txt'
Confidence(:,18) = Normalised(:,1);
load 'refrigirator/Normalised.txt'
Confidence(:,19) = Normalised(:,1);



plot(t,mean(Confidence,2))

% plot(t,Confidence)
% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')